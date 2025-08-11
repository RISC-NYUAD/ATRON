#include "op_solver_ros/op_solver_wrapper.h"
#include "op-solver.h"
#include "data/data.h"
#include "data/nearest/kdtree/kdtree.h"
#include "cp/cp.h"
#include "cp/heur/heur.h"
#include "cp/heur/ea/ea.h"
#include "op/op.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Function prototype
void cp_conf_heur_env(cp_prob *cp, cp_heur_env *heur_env);

// Define the global seed variable
#include <time.h>
unsigned long __seed__ = 0;

solver_data* op_wrapper_create_data() {
    return data_create();
}

void op_wrapper_free_data(solver_data** data) {
    data_free(data);
}

void op_wrapper_init_data(solver_data* data) {
    if (!data) {
        printf("ERROR: op_wrapper_init_data called with NULL data\n");
        return;
    }
    
    printf("op_wrapper_init_data: data->n = %d\n", data->n);
    
    // Set the distance calculation function based on norm type
    data_set_norm_type(data, data->norm);
    
    // Create initial map - even though OP problems don't use domain reduction,
    // the solver infrastructure requires a map to be present
    if (!data->map) {
        printf("op_wrapper_init_data: Creating map for %d nodes\n", data->n);
        // Allocate map structure
        data->map = (data_map*)malloc(sizeof(data_map));
        if (!data->map) {
            printf("ERROR: Failed to allocate map\n");
            return;
        }
        
        data_map* map = data->map;
        map->status = 1;
        map->img_n = data->n;
        map->dom_n = data->n;
        
        // Allocate arrays for identity mapping
        map->fun = (int*)malloc(data->n * sizeof(int));
        map->inv = (int*)malloc(data->n * sizeof(int));
        map->orig = (int*)malloc(data->n * sizeof(int));
        
        // Allocate coordinate arrays
        map->x = (double*)malloc(data->n * sizeof(double));
        map->y = (double*)malloc(data->n * sizeof(double));
        map->z = NULL;  // Not used for 2D problems
        map->w = NULL;  // Not used
        
        if (!map->fun || !map->inv || !map->orig || !map->x || !map->y) {
            // Cleanup on allocation failure
            free(map->fun);
            free(map->inv);
            free(map->orig);
            free(map->x);
            free(map->y);
            free(map);
            data->map = NULL;
            return;
        }
        
        // Initialize identity mapping and copy coordinates
        for (int i = 0; i < data->n; i++) {
            map->fun[i] = i;
            map->inv[i] = i;
            map->orig[i] = i;
            map->x[i] = data->x[i];
            map->y[i] = data->y[i];
        }
        
        // Initialize other fields
        map->kn_ecount = 0;
        map->kn_elist = NULL;
        map->kn_k = 0;
        map->prev = NULL;
        
        // Create kdtree for Euclidean problems
        if ((data->norm & SOLVER_DATA_TYPE_EUCLIDEAN) == SOLVER_DATA_TYPE_EUCLIDEAN) {
            map->kdtree = kdtree_create(data);
            printf("op_wrapper_init_data: Created kdtree for Euclidean problem\n");
        } else {
            map->kdtree = NULL;
        }
        
        printf("op_wrapper_init_data: Map created successfully, img_n = %d\n", map->img_n);
    } else {
        printf("op_wrapper_init_data: Map already exists\n");
    }
    
    // Initialize the distance cache
    data_create_cache(data);
    
    // Test distance calculation
    if (data->n >= 2) {
        double dist = data_get_norm(data, 0, 1);
        printf("op_wrapper_init_data: Test distance between nodes 0 and 1: %f\n", dist);
        
        // Also calculate manually to verify
        double dx = data->x[1] - data->x[0];
        double dy = data->y[1] - data->y[0];
        double manual_dist = sqrt(dx*dx + dy*dy);
        printf("op_wrapper_init_data: Manual distance calculation: %f\n", manual_dist);
        
        // Check the norm type
        printf("op_wrapper_init_data: data->norm = %d (EUCLIDEAN=%d)\n", data->norm, OP_WRAPPER_NORM_EUCLIDEAN);
        
        // Check if this is integer distance
        int int_dist = (int)dist;
        if ((double)int_dist == dist) {
            printf("op_wrapper_init_data: Distance appears to be integer-rounded\n");
        }
    }
    
    printf("op_wrapper_init_data: Initialization complete, data->map = %p\n", (void*)data->map);
}

void op_wrapper_set_data_name(solver_data* data, const char* name) {
    if (data && name) {
        strncpy(data->name, name, sizeof(data->name) - 1);
        data->name[sizeof(data->name) - 1] = '\0';
    }
}

void op_wrapper_set_data_prob(solver_data* data, int prob) {
    if (data) data->prob = prob;
}

void op_wrapper_set_data_norm(solver_data* data, int norm) {
    if (data) data->norm = norm;
}

void op_wrapper_set_data_n(solver_data* data, int n) {
    if (data) data->n = n;
}

void op_wrapper_allocate_arrays(solver_data* data) {
    if (!data || data->n <= 0) return;
    
    data->x = (double*)malloc(data->n * sizeof(double));
    data->y = (double*)malloc(data->n * sizeof(double));
    data->obj_node = (double*)malloc(data->n * sizeof(double));
}

void op_wrapper_set_data_coords(solver_data* data, int idx, double x, double y) {
    if (data && idx >= 0 && idx < data->n) {
        if (data->x) data->x[idx] = x;
        if (data->y) data->y[idx] = y;
    }
}

void op_wrapper_set_data_obj_node(solver_data* data, int idx, double obj) {
    if (data && idx >= 0 && idx < data->n && data->obj_node) {
        data->obj_node[idx] = obj;
    }
}

void op_wrapper_set_data_depot(solver_data* data, int from, int to) {
    if (data) {
        data->from = from;
        data->to = to;
    }
}

void op_wrapper_set_data_cap(solver_data* data, double cap) {
    if (data) data->cap = cap;
}

void op_wrapper_compute_total_obj(solver_data* data) {
    if (!data || !data->obj_node) return;
    
    double total = 0.0;
    for (int i = 0; i < data->n; i++) {
        total += data->obj_node[i];
    }
    data->tot_obj_node = total;
}

int op_wrapper_get_data_n(solver_data* data) {
    return data ? data->n : 0;
}

double op_wrapper_get_data_cap(solver_data* data) {
    return data ? data->cap : 0.0;
}

double op_wrapper_get_data_x(solver_data* data, int idx) {
    if (data && data->x && idx >= 0 && idx < data->n) {
        return data->x[idx];
    }
    return 0.0;
}

double op_wrapper_get_data_y(solver_data* data, int idx) {
    if (data && data->y && idx >= 0 && idx < data->n) {
        return data->y[idx];
    }
    return 0.0;
}

cp_env* op_wrapper_create_env() {
    // Initialize seed if not already done
    if (__seed__ == 0) {
        struct timespec tm_seed;
        clock_gettime(CLOCK_REALTIME, &tm_seed);
        __seed__ = tm_seed.tv_nsec / 1000;
        // Initialize random number generator
        srand(__seed__);
        srand48(__seed__);
    }
    
    cp_env* env = op_create_env();
    if (env && env->param) {
        env->param->appr = SOLVER_CP_APPR_HEUR_EA;  // Should be 2
        
        // Use default EA parameters (same as command-line tool)
        if (env->heur && env->heur->ea && env->heur->ea->param) {
            // Comment out custom parameters - use defaults for now
            // env->heur->ea->param->it_lim = 100;  
            // env->heur->ea->param->time_limit = 1000;  
            // env->heur->ea->param->pop_size = 20;  
            // env->heur->ea->param->pop_stop = 10;  
            printf("op_wrapper_create_env: EA params: it_lim=%d, time_limit=%ld, pop_size=%d\n",
                   env->heur->ea->param->it_lim, env->heur->ea->param->time_limit, 
                   env->heur->ea->param->pop_size);
        }
        
        // Set heuristic parameters
        if (env->heur && env->heur->param) {
            env->heur->param->init = SOLVER_CP_INIT_RAND;  // Use random initialization
            env->heur->param->improve_sol = SOLVER_CP_ADD_3N;  // Use 3-nearest improvement
        }
        
        // Parse minimal arguments to initialize the environment properly
        char *argv[] = {"op-solver", "opt", "--op-exact", "0", NULL};
        int argc = 4;
        cp_parse_args(argc, argv, env);
    }
    return env;
}

void op_wrapper_free_env(cp_env** env) {
    op_free_env(env);
}

cp_prob* op_wrapper_create_prob(solver_data* data) {
    printf("op_wrapper_create_prob: data=%p, data->n=%d\n", (void*)data, data ? data->n : -1);
    if (data && data->map) {
        printf("op_wrapper_create_prob: data->map=%p, data->map->img_n=%d\n", 
               (void*)data->map, data->map->img_n);
    }
    cp_prob* prob = op_create_prob(data);
    printf("op_wrapper_create_prob: created prob=%p\n", (void*)prob);
    if (prob) {
        printf("op_wrapper_create_prob: prob->n=%d, prob->sol=%p\n", prob->n, (void*)prob->sol);
    }
    return prob;
}

void op_wrapper_free_prob(cp_prob** prob) {
    op_free_prob(prob);
}

int op_wrapper_solve(cp_prob* prob, cp_env* env, cp_sol* sol) {
    printf("op_wrapper_solve: prob=%p, env=%p, sol=%p\n", (void*)prob, (void*)env, (void*)sol);
    printf("op_wrapper_solve: prob->n=%d, prob->cap=%f\n", prob->n, prob->cap);
    printf("op_wrapper_solve: prob->sol=%p\n", (void*)prob->sol);
    
    // Check if prob->sol is properly initialized
    if (!prob->sol) {
        printf("ERROR: prob->sol is NULL\n");
        return -1;
    }
    
    // Initialize the solution structure
    if (prob->sol->cycle == NULL) {
        printf("WARNING: prob->sol->cycle is NULL, this may cause issues\n");
    }
    
    // Configure heuristic environment for the problem
    if (env->heur) {
        cp_conf_heur_env(prob, env->heur);
        printf("op_wrapper_solve: Configured heuristic environment\n");
    }
    
    printf("op_wrapper_solve: Starting op_opt\n");
    printf("op_wrapper_solve: prob->from=%d, prob->to=%d\n", prob->from, prob->to);
    printf("op_wrapper_solve: env=%p, env->param=%p, env->heur=%p\n", 
           (void*)env, env ? (void*)env->param : NULL, env ? (void*)env->heur : NULL);
    if (env && env->param) {
        printf("op_wrapper_solve: env->param->appr=%d (expect %d for EA)\n", 
               env->param->appr, SOLVER_CP_APPR_HEUR_EA);
    }
    fflush(stdout);
    
    // op_opt expects to use prob->sol as both input and output
    int result = op_opt(prob, env, prob->sol);
    
    printf("op_wrapper_solve: op_opt returned %d\n", result);
    
    if (result == 0 && prob->sol) {
        printf("op_wrapper_solve: Solution status after solve: ns=%d, val=%f, cap=%f\n", 
               prob->sol->ns, prob->sol->val, prob->sol->cap);
        if (prob->sol->cycle) {
            printf("op_wrapper_solve: Solution cycle exists, first few nodes: ");
            for (int i = 0; i < prob->sol->ns && i < 5; i++) {
                printf("%d ", prob->sol->cycle[i]);
            }
            printf("\n");
        } else {
            printf("op_wrapper_solve: Solution cycle is NULL\n");
        }
    } else if (result == 0) {
        printf("op_wrapper_solve: Success but prob->sol is NULL\n");
    }
    
    return result;
}