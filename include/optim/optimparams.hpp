/*
    sfacpp - Estimation of TRE/GTRE SFA models
    Copyright (C) 2025 Edmund Haacke
    Copyright (C) 2025 NHS England

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/**
 * @file optimparams.hpp
 */

#ifndef ESA_OPTIM_PARAMS_HPP
#define ESA_OPTIM_PARAMS_HPP

#include <limits>
#include <cmath>

typedef struct ESAOptimParams {
    // maximum number of iterations
    unsigned int maxit = 450;
    // gradient error toleratance
    double grad_err_tol = 1e-08;
    double grad_err_tol_check = 1e-4;
    // dlib trust region
    double tr_radius = 1;
    // 
    double step1_grad_tol = 1e-2;
    // ---------------------------------------------------------------------
    // i think these can all be binned 
    // relative solution change tolerance - e.g., error tolerance controlling
    // how small the proportional change in the solution vector before convergence
    double rel_solution_change_err_tol = 1e-14;
    double rel_solution_change_err_tol_violations = 5; // cppoptlib specific
    // relative objective function change tolerance
    double rel_objfn_change_err_tol = 1e-08;
    double rel_objfn_change_err_tol_violations = 5; // cppoptlib specific
    
    //cppoptlib additional ones
    double condition_hessian = 0.0;
    double constraint_threshold = 1e-5;

    

    // line search tuning parameters
    double wolfe_cons_1 = 1e-3;
    double wolfe_cons_2 = 0.90;
    
    // setp size / learning rate
    double step_size = 0.1;

    // ---- Adam ---
    double adam_beta_1 = 0.9;
    double adam_beta_2 = 0.999;

    // ---- conjugate gradient ----
    bool use_rel_solution_change_crit = false;
    int cg_method = 2;
    double cg_restart_threshold = 0.1;

    // ---- gradient descent ----
    int gd_method = 0;
    // decay
    bool gd_step_decay = false;
    unsigned int gd_step_decay_periods = 10;
    double gd_step_decay_val = 0.5;
    // momentum parameters
    double gd_par_momentum = 0.9;
    // Ada params
    double gd_par_ada_norm_term = 1.0e-08;
    double gd_par_ada_rho = 0.9;
    bool gd_ada_max = false;
    // Adam parameters - see adam_beta_1, adam_beta_2;
    // gradient clippingg
    bool gd_clip_grad = false;
    bool gd_clip_max_norm = false;
    bool gd_clip_min_norm = false;
    int gd_clip_norm_type = 2;
    double gd_clip_norm_bound = 5.0;

    // ---- lbfgs ----
    size_t lbfgs_par_M = 10;

    // ---- nelder-mead ----
    bool nm_adaptive_pars = true;
    double nm_par_alpha = 1.0; // reflection parameter
    double nm_par_beta = 0.5; // contraction parameter
    double nm_par_gamma = 2.0; // expansion parameter
    double nm_par_delta = 0.5; // shrinkage parameters
    bool nm_custom_initial_simplex = false;


    // ---------
    unsigned int batch_size = 32;
    double init_mean_sq_grad_param = 1e-8;
    bool diff_seperable_shuffle = true;
    bool diff_seperable_reset_policy = true;
    bool diff_seperable_exact_obj = true;
    double momentum_decay = 0.5;

    // ---- ensmallen - SGDR ----
    unsigned int sgd_epoch_restart = 50;
    double sgd_batch_mult_factor = 2.0;
    
    // ---- ensmallen - simulated annealing ----
    double sa_init_temp = 10000.0; // initial temperature
    unsigned int sa_init_moves = 1000; // initial iterations w/o changing temp
    unsigned int sa_move_ctrl_sweep = 100; // sweeps/feedback move ctrl
    double sa_tol_frozen = 1e-5; // tolerance to consider system frozen
    unsigned int sa_max_tol_sweep = 3; // max sweeps below tolerance to consider frozen
    double sa_max_move_coef = 20.0; // maximum move size
    double sa_init_move_coef = 0.3; // initial move size
    double sa_gain = 0.3; // proportional control in feedback move ctrl

    // ---- PSO ----
    int pso_num_particles = 40;
    int pso_max_iter = 200;
    double pso_phi1 = 2.05;
    double pso_phi2 = 2.05;
    double pso_init_range = 2.0;   // half-width around starting values for focused stage
    double pso_broad_range = 5.0;  // half-width for broad fallback stage
    int pso_broad_max_iter = 300;  // iterations for broad fallback stage
    std::string pso_topology = "von_neumann";
    int pso_stagnation_patience = 50;
    double pso_stagnation_tol = 1e-8;
    double pso_vmax_fraction = 0.2;

    // step back
    double step_back = std::pow(std::numeric_limits<double>::epsilon(), 0.5);
    // relative tolerance
    double reltol = std::sqrt(std::numeric_limits<double>::epsilon());
    // limit tolerance
    double lmtol = std::sqrt(std::numeric_limits<double>::epsilon());
    // step tolerance
    double steptol = std::sqrt(std::numeric_limits<double>::epsilon());
    // when is classified as backed up
    double when_backedup = std::sqrt(std::numeric_limits<double>::epsilon());
    // how many times to allow for backedup status
    unsigned int max_backedup = 5;
    
    // whether or not to use analytical gradient
    bool useAnalyticalGradient = true;
    // seed to use for random number generator
    int seed = 1234;
    // pertubation
    double pertubation = 0.04;
    // epsilon
    double epsilon = 1e-8;
    // maximum number of backed up iterations allowed
    unsigned int max_total_backedup = 5;

    // ---- EM (Expectation-Maximization) parameters ----
    unsigned int em_max_iter     = 200;   // maximum EM outer iterations
    double       em_tol          = 1e-6;  // convergence: |LL^(t) - LL^(t-1)| < tol
    unsigned int em_nquad_pts    = 20;    // GHQ points (7, 10, 15, or 20)
    unsigned int em_seg_max_iter = 50;    // Newton iterations for seg M-step
    double       em_seg_tol      = 1e-8;  // convergence tolerance for seg M-step
    unsigned int em_class_max_iter = 450; // trust-region iterations per class M-step
    // ---- AGHQ mode-finding (Newton with finite differences) ----
    unsigned int em_aghq_newton_iter  = 10;   // Newton iterations to find posterior mode
    double       em_aghq_newton_delta = 1e-3; // finite-difference step size
} ESAOptimParams;

typedef struct ESAOptimParamsCppOpt {
    // maximum number of iterations
    unsigned int maxit = 450;
    // x delta
    double xdelta = 1e-9;
    unsigned int xdeltaViolate = 5;
    double fdelta = 1e-9;
    unsigned int fdeltaViolate = 5;
    double gradientNorm = 1e-6;
    double conditionHess = 0;
    double constraintThres = 1e-5;
} ESAOptimParamsCppOpt;

#endif // ESA_OPTIM_PARAMS_HPP