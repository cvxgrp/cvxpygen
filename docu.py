
import numpy as np


def replace_html_data(text, info_opt, info_cg, info_usr, info_can):
    """
    Replace placeholder strings in html documentation file
    """

    # code_dir
    text = text.replace('$CODEDIR', info_opt['code_dir'])
    text = text.replace('$CDPYTHON', info_opt['code_dir'].replace('/', '.').replace('\\', '.'))

    # solver name and docu
    text = text.replace('$CPGSOLVERNAME', info_opt['solver_name'])
    if info_opt['solver_name'] == 'OSQP':
        text = text.replace('$CPGSOLVERDOCUURL', 'https://osqp.org/docs/codegen/python.html')
    elif info_opt['solver_name'] == 'SCS':
        text = text.replace('$CPGSOLVERDOCUURL', 'https://www.cvxgrp.org/scs/api/c.html')
    elif info_opt['solver_name'] == 'ECOS':
        text = text.replace('$CPGSOLVERDOCUURL', 'https://github.com/embotech/ecos/wiki/Usage-from-C')

    # CMake prefix
    text = text.replace('$CPGCMAKELISTS', info_opt['prob_name']+'cpg')

    # basic type definitions
    if info_opt['solver_name'] in ['SCS', 'ECOS']:
        if info_opt['solver_name'] == 'SCS':
            CPGBASICTYPEDEF = '\ntypedef scs_float c_float;\n'
            CPGBASICTYPEDEF += 'typedef scs_int c_int;\n\n'
        else:
            CPGBASICTYPEDEF = '\ntypedef double c_float;\n'
            CPGBASICTYPEDEF += 'typedef int c_int;\n\n'
        CPGBASICTYPEDEF += '\n// Compressed sparse column (csc) matrix\n'
        CPGBASICTYPEDEF += 'typedef struct {\n'
        CPGBASICTYPEDEF += '  c_int      nzmax;\n'
        CPGBASICTYPEDEF += '  c_int      n;\n'
        CPGBASICTYPEDEF += '  c_int      m;\n'
        CPGBASICTYPEDEF += '  c_int      *p;\n'
        CPGBASICTYPEDEF += '  c_int      *i;\n'
        CPGBASICTYPEDEF += '  c_float    *x;\n'
        CPGBASICTYPEDEF += '  c_int      nz;\n'
        CPGBASICTYPEDEF += '} csc;\n'
        text = text.replace('$CPGBASICTYPEDEF', CPGBASICTYPEDEF)
    else:
        text = text.replace('$CPGBASICTYPEDEF\n', '')

    # type definition of CPG_Params_t or cpg_params_vec
    if info_opt['explicit']:
        CPGUSERPARAMSTYPEDEF = '\n// Struct type with user-defined parameters as fields\n'
        CPGUSERPARAMSTYPEDEF += 'typedef struct {\n'
        for name, size in info_usr['p_name_to_size_usp'].items():
            if size == 1:
                s = ''
            else:
                s = '*'
            CPGUSERPARAMSTYPEDEF += ('  c_float    %s   // Your parameter %s\n' % ((s+name+';').ljust(9), name))
        CPGUSERPARAMSTYPEDEF += '} CPG_Params_t;\n'
        text = text.replace('$CPGUSERPARAMSTYPEDEF', CPGUSERPARAMSTYPEDEF)
    else:
        text = text.replace('$CPGUSERPARAMSTYPEDEF\n', '')

    # type definition of Canon_Params_t
    CPGCANONPARAMSTYPEDEF = '\n// Struct type with canonical parameters as fields\n'
    CPGCANONPARAMSTYPEDEF += 'typedef struct {\n'
    for p_id in info_can['p_id_to_size'].keys():
        if p_id.isupper():
            CPGCANONPARAMSTYPEDEF += ('  csc        *%s   // Canonical parameter %s\n' % ((p_id+';').ljust(8), p_id))
        else:
            if p_id == 'd':
                s = ''
            else:
                s = '*'
            CPGCANONPARAMSTYPEDEF += \
                ('  c_float    %s   // Canonical parameter %s\n' % ((s+p_id+';').ljust(9), p_id))
    CPGCANONPARAMSTYPEDEF += '} Canon_Params_t;\n'
    text = text.replace('$CPGCANONPARAMSTYPEDEF', CPGCANONPARAMSTYPEDEF)

    # type definition of Canon_Outdated_t
    CPGOUTDATEDTYPEDEF = '\n// Struct type with booleans as fields, ' \
                         'indicating if respective canonical parameter is outdated\n'
    CPGOUTDATEDTYPEDEF += 'typedef struct {\n'
    for p_id in info_can['p_id_to_size'].keys():
        CPGOUTDATEDTYPEDEF += ('  int        %s    // Bool, if canonical parameter %s outdated\n'
                               % ((p_id+';').ljust(8), p_id))
    CPGOUTDATEDTYPEDEF += '} Canon_Outdated_t;\n'
    text = text.replace('$CPGOUTDATEDTYPEDEF', CPGOUTDATEDTYPEDEF)

    # type definition of CPG_Info_t
    CPGINFOTYPEDEF = '\n// Solver information\n'
    CPGINFOTYPEDEF += 'typedef struct {\n'
    CPGINFOTYPEDEF += '  c_float    obj_val;    // Objective function value\n'
    CPGINFOTYPEDEF += '  c_int      iter;       // Number of iterations\n'
    if info_opt['solver_name'] in ['OSQP', 'SCS']:
        CPGINFOTYPEDEF += '  char       *status;    // Solver status\n'
    elif info_opt['solver_name'] == 'ECOS':
        CPGINFOTYPEDEF += '  c_int      status;     // Solver status\n'
    CPGINFOTYPEDEF += '  c_float    pri_res;    // Primal residual\n'
    CPGINFOTYPEDEF += '  c_float    dua_res;    // Dual residual\n'
    CPGINFOTYPEDEF += '} CPG_Info_t;\n'
    text = text.replace('$CPGINFOTYPEDEF', CPGINFOTYPEDEF)

    # type definition of CPG_Result_t
    CPGRESULTTYPEDEF = '\n// Struct type with user-defined objective value and solution as fields\n'
    CPGRESULTTYPEDEF += 'typedef struct {\n'
    for name, size in info_usr['v_name_to_size'].items():
        if size == 1:
            s = ''
        else:
            s = '*'
        CPGRESULTTYPEDEF += ('  c_float    %s   // Your variable %s\n' % ((s + name + ';').ljust(9), name))
    CPGRESULTTYPEDEF += '  CPG_Info_t *info;      // Solver information\n'
    CPGRESULTTYPEDEF += '} CPG_Result_t;\n'
    text = text.replace('$CPGRESULTTYPEDEF', CPGRESULTTYPEDEF)

    if info_opt['solver_name'] == 'ECOS':
        CPGSETTINGSTYPEDEF = '\n// Solver settings\n'
        CPGSETTINGSTYPEDEF += 'typedef struct {\n'
        for name, typ in info_can['settings_names_to_type'].items():
            CPGSETTINGSTYPEDEF += '  %s%s;\n' % (typ.ljust(11), name)
        CPGSETTINGSTYPEDEF += '} Canon_Settings_t;\n'
        text = text.replace('$CPGSETTINGSTYPEDEF', CPGSETTINGSTYPEDEF)
    else:
        text = text.replace('$CPGSETTINGSTYPEDEF\n', '')

    # parameter delarations
    if info_opt['explicit']:
        CPGPARAMDECLARATIONS = '\n// User-defined parameters\n'
        for name, value in info_usr['p_writable'].items():
            if not np.isscalar(value):
                CPGPARAMDECLARATIONS += 'c_float %s[%d];\n' % (info_opt['prob_name'] + 'cpg_' + name, value.size)
        CPGPARAMDECLARATIONS += '\n\n// Struct containing all user-defined parameters\n'
        CPGPARAMDECLARATIONS += 'CPG_Params_t %sCPG_Params;\n' % info_opt['prob_name']
    else:
        CPGPARAMDECLARATIONS = '\n// Vector containing flattened user-defined parameters\n'
        CPGPARAMDECLARATIONS += 'c_float %scpg_params_vec[%d];\n' \
                                % (info_opt['prob_name'], np.sum(list(info_usr['p_name_to_size_usp'].values())) + 1)
        CPGPARAMDECLARATIONS += '\n\n// Sparse mappings from user-defined to canonical parameters\n'
        for p_id, mapping in zip(info_can['p_id_to_size'].keys(), info_can['mappings']):
            if mapping.nnz > 0:
                CPGPARAMDECLARATIONS += 'csc %scanon_%s_map;\n' % (info_opt['prob_name'], p_id)
    text = text.replace('$CPGPARAMDECLARATIONS', CPGPARAMDECLARATIONS)

    # canonical parameter declarations
    CPGCANONPARAMDECLARATIONS = '\n// Canonical parameters\n'
    for p_id, size in info_can['p_id_to_size'].items():
        if size > 0:
            if p_id.isupper():
                CPGCANONPARAMDECLARATIONS += 'csc %scanon_%s;\n' % (info_opt['prob_name'], p_id)
                if info_opt['solver_name'] == 'ECOS':
                    CPGCANONPARAMDECLARATIONS += 'csc %scanon_%s_ECOS;\n' % (info_opt['prob_name'], p_id)
            elif p_id != 'd':
                CPGCANONPARAMDECLARATIONS += 'c_float %scanon_%s[%d];\n' % (info_opt['prob_name'], p_id, size)
                if info_opt['solver_name'] == 'ECOS':
                    CPGCANONPARAMDECLARATIONS += 'c_float %scanon_%s_ECOS[%d];\n' % (info_opt['prob_name'], p_id, size)
    CPGCANONPARAMDECLARATIONS += '\n\n// Struct containing canonical parameters\n'
    CPGCANONPARAMDECLARATIONS += 'Canon_Params_t %sCanon_Params;\n' % info_opt['prob_name']
    if info_opt['solver_name'] == 'ECOS':
        CPGCANONPARAMDECLARATIONS += 'Canon_Params_t %sCanon_Params_ECOS;\n' % info_opt['prob_name']
    text = text.replace('$CPGCANONPARAMDECLARATIONS', CPGCANONPARAMDECLARATIONS)

    # outdated declarations
    CPGCANONOUTDATEDDECLARATION = '\n// Struct containing flags for outdated canonical parameters\n'
    CPGCANONOUTDATEDDECLARATION += 'Canon_Outdated_t %sCanon_Outdated;\n' % info_opt['prob_name']
    text = text.replace('$CPGCANONOUTDATEDDECLARATION', CPGCANONOUTDATEDDECLARATION)

    # variable declarations
    CPGVARIABLEDECLARATIONS = '\n// User-defined variables\n'
    for name, size in info_usr['v_name_to_size'].items():
        if size > 1:
            CPGVARIABLEDECLARATIONS += 'c_float %s[%d];\n' % (info_opt['prob_name'] + 'cpg_' + name, size)
    text = text.replace('$CPGVARIABLEDECLARATIONS', CPGVARIABLEDECLARATIONS)

    # solver info and result declarations
    CPGINFORESULTDECLARATION = '\n// Struct containing solver info\n'
    CPGINFORESULTDECLARATION += 'CPG_Info_t %sCPG_Info;\n\n' % info_opt['prob_name']
    CPGINFORESULTDECLARATION += '\n// Struct containing user-defined objective value and solution\n'
    CPGINFORESULTDECLARATION += 'CPG_Result_t %sCPG_Result;\n' % info_opt['prob_name']
    text = text.replace('$CPGINFORESULTDECLARATION', CPGINFORESULTDECLARATION)

    # extra declarations
    if info_opt['solver_name'] == 'SCS':
        CPGEXTRADECLARATIONS = '\n// SCS matrix A\n'
        CPGEXTRADECLARATIONS += 'ScsMatrix %sscs_A;\n\n' % info_opt['prob_name']
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS data\n'
        CPGEXTRADECLARATIONS += 'ScsData %sScs_D;\n\n' % info_opt['prob_name']
        if info_can['constants']['qsize'] > 0:
            CPGEXTRADECLARATIONS += '\n// SCS array of SOC dimensions\n'
            CPGEXTRADECLARATIONS += 'c_int %sscs_q[%d];\n\n' % (info_opt['prob_name'], info_can['constants']['qsize'])
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS cone data\n'
        CPGEXTRADECLARATIONS += 'ScsCone %sScs_K;\n\n' % info_opt['prob_name']
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS settings\n'
        CPGEXTRADECLARATIONS += 'ScsSettings %sScs_Stgs;\n\n' % info_opt['prob_name']
        CPGEXTRADECLARATIONS += '\n// SCS solution\n'
        CPGEXTRADECLARATIONS += 'c_float %sscs_x[%d];\n' % (info_opt['prob_name'], info_can['constants']['n'])
        CPGEXTRADECLARATIONS += 'c_float %sscs_y[%d];\n' % (info_opt['prob_name'], info_can['constants']['m'])
        CPGEXTRADECLARATIONS += 'c_float %sscs_s[%d];\n\n' % (info_opt['prob_name'], info_can['constants']['m'])
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS solution\n'
        CPGEXTRADECLARATIONS += 'ScsSolution %sScs_Sol;\n\n' % info_opt['prob_name']
        CPGEXTRADECLARATIONS += '\n// Struct containing SCS information\n'
        CPGEXTRADECLARATIONS += 'ScsInfo %sScs_Info;\n' % info_opt['prob_name']
        text = text.replace('$CPGEXTRADECLARATIONS', CPGEXTRADECLARATIONS)
    elif info_opt['solver_name'] == 'ECOS':
        CPGEXTRADECLARATIONS = '\n// Struct containing solver settings\n'
        CPGEXTRADECLARATIONS += 'Canon_Settings_t %sCanon_Settings;\n\n' % info_opt['prob_name']
        if info_can['constants']['n_cones'] > 0:
            CPGEXTRADECLARATIONS += '\n// ECOS array of SOC dimensions\n'
            CPGEXTRADECLARATIONS += 'c_int %secos_q[%d];\n\n' % (info_opt['prob_name'],
                                                                 info_can['constants']['n_cones'])
        CPGEXTRADECLARATIONS += '\n// ECOS workspace\n'
        CPGEXTRADECLARATIONS += 'pwork* %secos_workspace;\n\n' % info_opt['prob_name']
        CPGEXTRADECLARATIONS += '\n// ECOS exit flag\n'
        CPGEXTRADECLARATIONS += 'c_int %secos_flag;\n' % info_opt['prob_name']
        text = text.replace('$CPGEXTRADECLARATIONS', CPGEXTRADECLARATIONS)
    else:
        text = text.replace('$CPGEXTRADECLARATIONS\n', '')

    # update declarations
    CPGUPDATEDECLARATIONS = '\n// Update user-defined parameter values\n'
    for name, size in info_usr['p_name_to_size_usp'].items():
        if size == 1:
            CPGUPDATEDECLARATIONS += 'void %scpg_update_%s(c_float value);\n' % (info_opt['prob_name'], name)
        else:
            CPGUPDATEDECLARATIONS += 'void %scpg_update_%s(c_int idx, c_float value);\n' % (info_opt['prob_name'], name)
    text = text.replace('$CPGUPDATEDECLARATIONS', CPGUPDATEDECLARATIONS)

    # canonicalize declarations
    CPGCANONICALIZEDECLARATIONS = '\n// Map user-defined to canonical parameters\n'
    for p_id in info_can['p_id_to_size'].keys():
        CPGCANONICALIZEDECLARATIONS += 'void %scpg_canonicalize_%s();\n' % (info_opt['prob_name'], p_id)
    text = text.replace('$CPGCANONICALIZEDECLARATIONS', CPGCANONICALIZEDECLARATIONS)

    # retrieve and solve declarations
    CPGRETRIEVESOLVEDECLARATIONS = ''
    if info_cg['ret_sol_func_exists']:
        CPGRETRIEVESOLVEDECLARATIONS += '\n// Retrieve solution in terms of user-defined variables\n'
        CPGRETRIEVESOLVEDECLARATIONS += 'void %scpg_retrieve_solution();\n\n' % info_opt['prob_name']
    CPGRETRIEVESOLVEDECLARATIONS += '\n// Retrieve solver information\n'
    CPGRETRIEVESOLVEDECLARATIONS += 'void %scpg_retrieve_info();\n\n' % info_opt['prob_name']
    CPGRETRIEVESOLVEDECLARATIONS += '\n// Solve via canonicalization, canonical solve, retrieval\n'
    CPGRETRIEVESOLVEDECLARATIONS += 'void %scpg_solve();\n' % info_opt['prob_name']
    text = text.replace('$CPGRETRIEVESOLVEDECLARATIONS', CPGRETRIEVESOLVEDECLARATIONS)

    # settings declarations
    CPGSETTINGSDECLARATIONS = '\n// Update solver settings\n'
    CPGSETTINGSDECLARATIONS += 'void %scpg_set_solver_default_settings();\n' % info_opt['prob_name']
    CPGSETTINGSDECLARATIONS += 'void %scpg_set_solver_&lt;setting_name&gt;' \
                               '(&lt;setting_type&gt;, &lt;setting_name&gt;_new);\n' % info_opt['prob_name']
    CPGSETTINGSDECLARATIONS += '...\n'

    return text.replace('$CPGSETTINGSDECLARATIONS', CPGSETTINGSDECLARATIONS)
