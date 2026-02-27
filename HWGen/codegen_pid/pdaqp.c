#include "pdaqp.h"
c_float_store pdaqp_halfplanes[6] = {
(c_float_store)0.5132947022251544,
(c_float_store)0.8582124146547812,
(c_float_store)1.283236755562886,
(c_float_store)0.7071067811865475,
(c_float_store)-0.7071067811865475,
(c_float_store)-0.7463904912524668,
};
c_float_store pdaqp_feedbacks[27] = {
(c_float_store)0.140625,
(c_float_store)-0.140625,
(c_float_store)0.1484375,
(c_float_store)-0.005208333333333315,
(c_float_store)0.005208333333333259,
(c_float_store)0.5130208333333334,
(c_float_store)-0.1354166666666667,
(c_float_store)0.13541666666666669,
(c_float_store)0.3385416666666668,
(c_float_store)0.19999999999999998,
(c_float_store)-0.04135188866799204,
(c_float_store)0.0,
(c_float_store)0.2,
(c_float_store)0.3483101391650099,
(c_float_store)0.0,
(c_float_store)9.656512391123383e-18,
(c_float_store)0.36182902584493054,
(c_float_store)0.0,
(c_float_store)0.0,
(c_float_store)5.551115123125783e-17,
(c_float_store)-3.071263974827799e-17,
(c_float_store)0.09523809523809522,
(c_float_store)-0.0952380952380954,
(c_float_store)0.6190476190476192,
(c_float_store)-0.09523809523809526,
(c_float_store)0.09523809523809523,
(c_float_store)0.38095238095238115,
};
c_int pdaqp_hp_list[5] = {
(c_int)1,
(c_int)0,
(c_int)2,
(c_int)1,
(c_int)0,
};
c_int pdaqp_jump_list[5] = {
(c_int)1,
(c_int)2,
(c_int)0,
(c_int)0,
(c_int)0,
};
void pdaqp_pid_evaluate(c_float* parameter, c_float* solution){
    int i,j,disp;
    int id,next_id;
    c_float val;
    id = 0;
    next_id = id+pdaqp_jump_list[id];
    while(next_id != id){
        // Compute halfplane value
        disp = pdaqp_hp_list[id]*(PDAQP_N_PARAMETER+1);
        for(i=0, val=0; i<PDAQP_N_PARAMETER; i++)
            val += parameter[i] * pdaqp_halfplanes[disp++];
        id = next_id + (val <= pdaqp_halfplanes[disp]);
        next_id = id+pdaqp_jump_list[id];
    }
    // Leaf node reached -> evaluate affine function
    disp = pdaqp_hp_list[id]*(PDAQP_N_PARAMETER+1)*PDAQP_N_SOLUTION;
    for(i=0; i < PDAQP_N_SOLUTION; i++){
        for(j=0, val=0; j < PDAQP_N_PARAMETER; j++)
            val += parameter[j] * pdaqp_feedbacks[disp++];
        val += pdaqp_feedbacks[disp++];
        solution[i] = val;
    }
}
