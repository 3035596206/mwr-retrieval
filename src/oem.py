"""OEM / 1D-Var solver: Levenberg-Marquardt Gauss-Newton iteration.

Implements the core OEM retrieval loop per Plan §10:
  Find x that minimises J(x) = (x-x_a)ᵀ S_a⁻¹ (x-x_a) + (y-H(x))ᵀ S_e⁻¹ (y-H(x))

Usage:
    from oem import OEMSolver

    solver = OEMSolver(forward_model, state_packer)
    result = solver.retrieve(y_obs, x_a, S_a, S_e)
"""

import numpy as np
import config


class OEMSolver:
    """Levenberg-Marquardt Gauss-Newton OEM solver.

    Iteratively solves for the optimal state vector x given:
      - y_obs: observed brightness temperatures
      - x_a:   background (a priori) state vector
      - S_a:   background error covariance
      - S_e:   observation error covariance
      - H(x):  forward model (via forward_model.simulate)
    """

    def __init__(self, forward_model, state_packer):
        """Initialise the solver.

        Args:
            forward_model: ForwardModel instance with .simulate(profile) method
            state_packer: OEMStatePacker instance for x ↔ profile mapping
        """
        self.H = forward_model
        self.packer = state_packer
        self.n_channels = forward_model.n_channels
        self.n_state = state_packer.n_state

    # ================================================================
    # Main retrieval entry point
    # ================================================================

    def retrieve(self, y_obs, x_a, S_a, S_e,
                 max_iter=15, cost_tol=1e-3, dx_tol=1e-4,
                 gamma_init=1.0, gamma_factor_down=0.5, gamma_factor_up=3.0,
                 adaptive_se=False, verbose=False):
        """Run the LM Gauss-Newton retrieval.

        Args:
            y_obs: observed brightness temperatures (n_channels,)
            x_a:   a priori state vector (n_state,)
            S_a:   background error covariance (n_state, n_state)
            S_e:   observation error covariance (n_channels, n_channels)
            max_iter: maximum iterations (default 15)
            cost_tol: convergence threshold on relative cost change
            dx_tol: convergence threshold on state update norm
            gamma_init: initial LM damping factor
            gamma_factor_down: gamma multiplier on successful step
            gamma_factor_up: gamma multiplier on rejected step
            adaptive_se: if True, recompute S_e from current LWC state each
                         iteration (cloud-adaptive error inflation, design doc §3.4)
            verbose: print iteration diagnostics

        Returns:
            result: dict with keys:
                x_retrieved          — final state vector
                x_background         — a priori state vector
                y_obs                — observed BT
                y_sim_background     — H(x_a)
                y_sim_retrieved      — H(x_retrieved)
                cost_history         — list of J_total per iteration
                converged            — bool
                n_iter               — number of iterations
                jacobian             — K at final state (if compute_diagnostics)
                averaging_kernel     — A = G K
                posterior_covariance — S_post
                dofs                 — trace(A)
                exit_reason          — string
        """
        from oem_covariance import compute_iclwc_from_profile, adapt_se_for_cloud

        Se_base = S_e.copy() if adaptive_se else S_e

        # Pre-compute inverses
        def _update_se_inv(profile):
            """Recompute Se_inv with cloud adaptation if enabled."""
            if not adaptive_se:
                return self._safe_inv(Se_base, "S_e")
            iclwc = compute_iclwc_from_profile(profile)
            Se_adapted = adapt_se_for_cloud(Se_base, iclwc)
            return self._safe_inv(Se_adapted, "S_e")

        Sa_inv = self._safe_inv(S_a, "S_a")

        # Initial state and forward
        x = x_a.copy()
        profile = self.packer.unpack(x)
        y_sim = self.H.simulate(profile)
        Se_inv = _update_se_inv(profile)

        # Initial cost
        dx_bg = x - x_a
        dy = y_obs - y_sim
        J_bg = float(dx_bg @ Sa_inv @ dx_bg)
        J_obs = float(dy @ Se_inv @ dy)
        J = J_bg + J_obs

        cost_history = [J]
        gamma = gamma_init

        if verbose:
            print(f"[OEM] iter=0  J_total={J:.4f}  J_bg={J_bg:.4f}  J_obs={J_obs:.4f}  gamma={gamma:.2e}")

        converged = False
        exit_reason = "max_iter"
        K = None

        for iteration in range(1, max_iter + 1):
            # ---- compute Jacobian at current state ----
            K = self.H.jacobian(x, self.packer)

            # ---- Gauss-Newton step with LM damping ----
            # Δx = [Kᵀ S_e⁻¹ K + S_a⁻¹ + γI]⁻¹ [Kᵀ S_e⁻¹ (y - H(x)) - S_a⁻¹ (x - x_a)]
            Kt_Se_inv = K.T @ Se_inv                    # (n_state, n_channels)
            Hessian = Kt_Se_inv @ K + Sa_inv + gamma * np.eye(self.n_state)
            grad = Kt_Se_inv @ dy - Sa_inv @ dx_bg

            try:
                dx = np.linalg.solve(Hessian, grad)
            except np.linalg.LinAlgError:
                if verbose:
                    print(f"[OEM] iter={iteration}  Hessian singular, increasing gamma")
                gamma *= gamma_factor_up
                continue

            # ---- trial state ----
            x_trial = x + dx
            profile_trial = self.packer.unpack(x_trial)
            y_trial = self.H.simulate(profile_trial)

            dx_trial_bg = x_trial - x_a
            dy_trial = y_obs - y_trial
            J_trial_bg = float(dx_trial_bg @ Sa_inv @ dx_trial_bg)
            J_trial_obs = float(dy_trial @ Se_inv @ dy_trial)
            J_trial = J_trial_bg + J_trial_obs

            # ---- LM acceptance test ----
            if J_trial < J:
                # Accept step
                x = x_trial
                y_sim = y_trial
                dx_bg = dx_trial_bg
                dy = dy_trial
                J_new = J_trial
                gamma *= gamma_factor_down
                gamma = max(gamma, 1e-6)

                # Update adaptive Se_inv from new profile state
                if adaptive_se:
                    profile_current = self.packer.unpack(x)
                    Se_inv = _update_se_inv(profile_current)

                if verbose:
                    print(f"[OEM] iter={iteration}  J_total={J_new:.4f}  J_bg={J_trial_bg:.4f}  J_obs={J_trial_obs:.4f}  gamma={gamma:.2e}  accept")

                # Check convergence
                rel_change = abs(J - J_new) / (abs(J) + 1e-10)
                J = J_new
                cost_history.append(J)

                if rel_change < cost_tol:
                    converged = True
                    exit_reason = f"cost_converged (rel_change={rel_change:.2e})"
                    if verbose:
                        print(f"[OEM] Converged: rel_cost_change={rel_change:.2e} < {cost_tol}")
                    break

                if np.linalg.norm(dx) < dx_tol * (np.linalg.norm(x) + 1e-10):
                    converged = True
                    exit_reason = f"dx_converged (|dx|={np.linalg.norm(dx):.2e})"
                    if verbose:
                        print(f"[OEM] Converged: |dx|={np.linalg.norm(dx):.2e} < {dx_tol}")
                    break

            else:
                # Reject step, increase damping
                gamma *= gamma_factor_up
                cost_history.append(J)  # cost unchanged

                if verbose:
                    print(f"[OEM] iter={iteration}  J_trial={J_trial:.4f} > J={J:.4f}  gamma->{gamma:.2e}  reject")

                if gamma > 1e6:
                    exit_reason = "damping_exhausted"
                    if verbose:
                        print("[OEM] Damping factor exhausted, stopping.")
                    break

        # ---- final Jacobian (recompute if not already at final state) ----
        if K is None or converged:
            K = self.H.jacobian(x, self.packer)

        # ---- diagnostics ----
        G = self._gain_matrix(K, Sa_inv, Se_inv)
        A = G @ K                                      # averaging kernel
        S_post = self._posterior_covariance(K, Sa_inv, Se_inv)
        dofs = float(np.trace(A))

        # ---- build result ----
        result = {
            "x_retrieved": x,
            "x_background": x_a,
            "y_obs": y_obs,
            "y_sim_background": self._compute_y_background(x_a),
            "y_sim_retrieved": y_sim,
            "cost_history": np.array(cost_history),
            "converged": converged,
            "n_iter": iteration if converged else max_iter,
            "jacobian": K,
            "averaging_kernel": A,
            "posterior_covariance": S_post,
            "dofs": dofs,
            "exit_reason": exit_reason,
        }

        # Add cloud diagnostic if adaptive S_e was used
        if adaptive_se:
            from oem_covariance import compute_iclwc_from_profile
            result["iclwc_final"] = compute_iclwc_from_profile(
                self.packer.unpack(x)
            )

        return result

    # ================================================================
    # Diagnostic helpers
    # ================================================================

    def _compute_y_background(self, x_a):
        """Compute H(x_a) for the background state."""
        profile_bg = self.packer.unpack(x_a)
        return self.H.simulate(profile_bg)

    def _gain_matrix(self, K, Sa_inv, Se_inv):
        """G = [Kᵀ S_e⁻¹ K + S_a⁻¹]⁻¹ Kᵀ S_e⁻¹"""
        Hess = K.T @ Se_inv @ K + Sa_inv
        try:
            Hess_inv = np.linalg.inv(Hess)
        except np.linalg.LinAlgError:
            Hess_inv = np.linalg.pinv(Hess)
        return Hess_inv @ K.T @ Se_inv

    def _posterior_covariance(self, K, Sa_inv, Se_inv):
        """S_post = [Kᵀ S_e⁻¹ K + S_a⁻¹]⁻¹"""
        Hess = K.T @ Se_inv @ K + Sa_inv
        try:
            return np.linalg.inv(Hess)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(Hess)

    @staticmethod
    def _safe_inv(M, name="matrix"):
        """Invert a matrix with fallback to pseudo-inverse."""
        try:
            return np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print(f"[OEM] Warning: {name} is singular, using pinv.")
            return np.linalg.pinv(M)


# ================================================================
# Convenience: compute retrieval profile(s) from result
# ================================================================

def retrieval_profile(result, state_packer):
    """Unpack retrieval result into a physical profile dict.

    Args:
        result: dict returned by OEMSolver.retrieve()
        state_packer: OEMStatePacker instance

    Returns:
        profile: dict with T, RH, CLWC, height
    """
    return state_packer.unpack(result["x_retrieved"])


def background_profile(result, state_packer):
    """Unpack background into a physical profile dict."""
    return state_packer.unpack(result["x_background"])
