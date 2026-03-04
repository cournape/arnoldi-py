import numpy as np


class Callback:
    # Called at the very beginning of the outer loop
    def on_arnoldi_start(self, restart, A, V, H, start_dim, max_dim):
        pass

    # Called after arnoldi decomposition is done
    def on_arnoldi_end(self, restart, A, V, H, active_dim):
        pass

    def on_convergence_check(self, restart, A, ritz_values, beta_m, q_m) -> np.ndarray:
        """ Called just before Schur truncation, to check convergence.

        beta_m and q_m are vectors defined from the Arnoldi relationship:

            A V_m = V_m H + beta_m * v_m * e_m^T

        with e_m the basis vector for the m'th dimension. After Schur transform
        H = Q T Q^H

            A V_m Q = V_m Q T + beta_m * v_m * q_m^H

        And as v_m norm is 1 by Arnoldi construation, the residuals are:

            ||A V_m Q - V_m Q T|| = |beta_m| * ||q_m||

        Parameters
        ==========
        restart : int
            The iteration number
        A : array-like
            The operator to solve the eigen problem for
        ritz_values : np.ndarray
            Same shape as approximate_residuals. The diagonal values of the T
            from the (reordered) Schur transform Q T Q^H computed from Arnoldi
            decomposition. Reordered means that the ritz_values are ordered
            following the desired final order
        beta_m : float
            Coupling factor (aka H_a[-1, -1] where H_a is the active
        approximate_residuals : np.ndarray
            The approximate_residuals as computed from Arnoldi decomposition,
            after rotation from the Schur Transform

        Returns
        =======
        errors : np.ndarray
            The errors estimate that will be tested against tolerance. Should
            have the same shape as approximate residuals/ritz values
        """
        approximate_residuals = np.abs(beta_m * q_m)
        return approximate_residuals / np.abs(ritz_values)

    def on_restart_end(self, restart, n_converged, ritz_values, errors):
        """ Called at the end of the restart loop.

        Parameters
        ==========
        restart : int
            The iteration number
        n_converged : int
            Number of converged values
        ritz_values : np.ndarray
            The ritz values
        errors : np.ndarray
            The errors as used for the convergence criteria
        """
