def build_var_groups(tv_iphi_vars, tv_iphi_offsetl, tv_iphi_offsett, tv_iphi_const, tv_ophi_vars, tv_ophi_offsett_ophi): ...
def get_var_info(var, var_groups): ...
def build_linear_constraints(x_len, index_pairs_levels, index_pairs_times, var_groups): ...
def constraint_violation(x, A_level, lb_level, A_time, lb_time): ...
def penalized_objective(x, obj_args, constraint_args, penalty_weight: float = 10000.0): ...
