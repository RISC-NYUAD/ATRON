#ifndef OP_SOLVER_WRAPPER_H
#define OP_SOLVER_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations to avoid including GMP headers in C++ code
typedef struct solver_data solver_data;
typedef struct cp_prob cp_prob;
typedef struct cp_env cp_env;
typedef struct cp_sol cp_sol;

// Wrapper functions that don't expose GMP types
solver_data* op_wrapper_create_data();
void op_wrapper_free_data(solver_data** data);
void op_wrapper_init_data(solver_data* data);  // Initialize map and cache

// Data setters
void op_wrapper_set_data_name(solver_data* data, const char* name);
void op_wrapper_set_data_prob(solver_data* data, int prob);
void op_wrapper_set_data_norm(solver_data* data, int norm);
void op_wrapper_set_data_n(solver_data* data, int n);
void op_wrapper_set_data_coords(solver_data* data, int idx, double x, double y);
void op_wrapper_set_data_obj_node(solver_data* data, int idx, double obj);
void op_wrapper_set_data_depot(solver_data* data, int from, int to);
void op_wrapper_set_data_cap(solver_data* data, double cap);
void op_wrapper_allocate_arrays(solver_data* data);
void op_wrapper_compute_total_obj(solver_data* data);

// Data getters
int op_wrapper_get_data_n(solver_data* data);
double op_wrapper_get_data_cap(solver_data* data);
double op_wrapper_get_data_x(solver_data* data, int idx);
double op_wrapper_get_data_y(solver_data* data, int idx);

cp_env* op_wrapper_create_env();
void op_wrapper_free_env(cp_env** env);
cp_prob* op_wrapper_create_prob(solver_data* data);
void op_wrapper_free_prob(cp_prob** prob);
int op_wrapper_solve(cp_prob* prob, cp_env* env, cp_sol* sol);

// Constants - matching op-solver internal values
#define OP_WRAPPER_PROB_OP 1  // SOLVER_PROB_OP from data.h
#define OP_WRAPPER_NORM_EUCLIDEAN 1154  // SOLVER_DATA_NORM_EUCLIDEAN = (2 | 128 | 1024)
#define OP_WRAPPER_APPR_HEUR_EA 1
#define SOLVER_CP_APPR_HEUR_EA 2  // From cp/cp.h

// Heuristic initialization methods
#define SOLVER_CP_INIT_RAND 2
#define SOLVER_CP_INIT_BEST3 0

// Heuristic improvement methods  
#define SOLVER_CP_ADD_3N 4
#define SOLVER_CP_ADD_IN 5

#ifdef __cplusplus
}
#endif

#endif // OP_SOLVER_WRAPPER_H