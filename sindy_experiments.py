import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

import tellurium as te

import basis_functions

def run_sindy(data, param_names, d=1, time_steps=1, alph=1e-1, plot=False,
        alpha=1e-2, fit_intercept=True):
    # alph and alpha are two separate parameters for regularization in two different contexts
    # alph regularizes the derivative calculation
    # alpha regularizes the Lasso
    """
    Runs sindy to generate a system of ODEs that represent the given data...
    """
    basis_vectors, all_combs = basis_functions.generate_basis_functions(data, d=d)
    param_names = np.array(param_names)
    all_var_combs = [param_names[list(c)].tolist() for c in all_combs]
    derivs = basis_functions.calculate_derivatives(data, regularize=True,
            time_steps=time_steps, alph=alph)
    if plot:
        plt.title('derivs')
        plt.plot(derivs)
        plt.show()
    model = Lasso(fit_intercept=fit_intercept, alpha=alpha)
    model.fit(basis_vectors, derivs[1:,:])
    params = model.coef_
    intercept = model.intercept_
    # calculate that one thing...
    nonzero_params = np.nonzero(params)
    param_equations = {v:[] for v in param_names}
    for x, y in zip(*nonzero_params):
        print(x, y)
        print(all_var_combs[y])
        print(param_names[x])
        print(params[x, y])
        param_equations[param_names[x]].append((all_var_combs[y], params[x,y]))
    index = 0
    antimony_eqs = ''
    antimony_vars = ''
    param_index = 0
    for var in sorted(param_names):
        params = param_equations[var]
        positive_params = [p for p in params if p[1] > 0]
        positive_rate = ' + '.join(str(x[1]) + '*' + '*'.join(x[0]) for x in positive_params)

        negative_params = [p for p in params if p[1] < 0]
        negative_rate = ' + '.join(str(-x[1]) + '*' + '*'.join(x[0]) for x in negative_params)
        try:
            if intercept[index] > 0:
                if len(positive_rate) == 0:
                    positive_rate = str(intercept[index])
                else:
                    positive_rate = ' + '.join([str(intercept[index]), positive_rate])
            else:
                if len(negative_rate) == 0:
                    negative_rate = str(-intercept[index])
                else:
                    negative_rate = ' + '.join([str(-intercept[index]), negative_rate])
        except:
            pass

        antimony_eqs += '\nJ{0}: -> {1}; {2}'.format(index*2, var, positive_rate)
        antimony_eqs += '\nJ{0}: {1} ->; {2}'.format(index*2+1, var, negative_rate)
        index += 1
    print(antimony_eqs)
    return antimony_eqs

def run_implicit_sindy(data, param_names, d=1, time_steps=1, alph=1e-1, plot=False,
        alpha=1e-2, fit_intercept=True):
    # alph and alpha are two separate parameters for regularization in two different contexts
    # alph regularizes the derivative calculation
    # alpha regularizes the Lasso
    """
    Runs implicit sindy to generate a system of ODEs that represent the given data...
    in this implementation, we use polynomial basis functions
    """
    # TODO
    basis_vectors, all_combs = basis_functions.generate_basis_functions(data, d=d)
    param_names = np.array(param_names)
    all_var_combs = [param_names[list(c)].tolist() for c in all_combs]
    derivs = basis_functions.calculate_derivatives(data, regularize=True,
            time_steps=time_steps, alph=alph)
    if plot:
        plt.title('derivs')
        plt.plot(derivs)
        plt.show()
    model = Lasso(fit_intercept=fit_intercept, alpha=alpha)
    model.fit(basis_vectors, derivs[1:,:])
    params = model.coef_
    intercept = model.intercept_
    # calculate that one thing...
    nonzero_params = np.nonzero(params)
    param_equations = {v:[] for v in param_names}
    for x, y in zip(*nonzero_params):
        print(x, y)
        print(all_var_combs[y])
        print(param_names[x])
        print(params[x, y])
        param_equations[param_names[x]].append((all_var_combs[y], params[x,y]))
    index = 0
    antimony_eqs = ''
    antimony_vars = ''
    param_index = 0
    for var in sorted(param_names):
        params = param_equations[var]
        positive_params = [p for p in params if p[1] > 0]
        positive_rate = ' + '.join(str(x[1]) + '*' + '*'.join(x[0]) for x in positive_params)

        negative_params = [p for p in params if p[1] < 0]
        negative_rate = ' + '.join(str(-x[1]) + '*' + '*'.join(x[0]) for x in negative_params)
        try:
            if intercept[index] > 0:
                if len(positive_rate) == 0:
                    positive_rate = str(intercept[index])
                else:
                    positive_rate = ' + '.join([str(intercept[index]), positive_rate])
            else:
                if len(negative_rate) == 0:
                    negative_rate = str(-intercept[index])
                else:
                    negative_rate = ' + '.join([str(-intercept[index]), negative_rate])
        except:
            pass

        antimony_eqs += '\nJ{0}: -> {1}; {2}'.format(index*2, var, positive_rate)
        antimony_eqs += '\nJ{0}: {1} ->; {2}'.format(index*2+1, var, negative_rate)
        index += 1
    print(antimony_eqs)
    return antimony_eqs



if __name__ == '__main__':
    # test 1: 3-node model from class on 2018-10-26
    model1 = te.loada("""
            J1: -> A ; v0
            J2: A -> B ; ka*A
            J3: B -> C ; kb*B
            J4: C -> ; kc*C

            v0 = 10;
            ka = 0.4;
            kb = 0.32;
            kc = 0.4;
            """)
    results = model1.simulate(0, 50, 50)
    data = np.column_stack([results['[A]'], results['[B]'], results['[C]']])

    param_names = ['A', 'B', 'C']
    noise_variance = 1

    # add noise to simulated model
    rmse_vals = {}
    for alpha in [1, 1e-1, 1e-2, 1e-3]:
        for alph in [10, 1, 1e-1, 1e-2, 1e-3]:
            for iter in range(10):
                data = np.column_stack([results['[A]'], results['[B]'], results['[C]']])
                data_deriv = data[1:,:] - data[:-1,:]
                noise = np.random.randn(*data.shape)*noise_variance
                data += noise
                estimated_deriv = basis_functions.calculate_derivatives(data, regularize=True, alph=alph)
                print('Total SNR:', np.mean(data)/np.std(data))
                for i in range(data.shape[1]):
                    print('SNR for {0}:'.format(param_names[i]), np.mean(data[:,i])/np.std(data[:,i]))

                #plt.plot(results['time'], data[:,0])
                #plt.plot(results['time'], data[:,1])
                #plt.plot(results['time'], data[:,2])
                #plt.show()

                # use sindy to generate a new antimony model
                antimony_eqs = run_sindy(data, param_names, d=1, alpha=alpha, alph=alph, plot=True, fit_intercept=True)
                model2 = te.loada(antimony_eqs)
                results_2 = model2.simulate(0, 50, 50)
                data_2 = np.column_stack([results_2['[A]'], results_2['[B]'], results_2['[C]']])

                error = np.sqrt(np.mean((data_2 - data)**2))
                print('RMSE:', error)
                key = str((alpha, alph))
                if key not in rmse_vals:
                    rmse_vals[key] = [error]
                else:
                    rmse_vals[key].append(error)

    alpha_vals = []
    alph_vals = []
    errors = []
    for key, val in rmse_vals.items():
        alpha = float(key.split(',')[0][1:])
        alph = float(key.split(',')[1][:-1])
        for error in val:
            alpha_vals.append(alpha)
            alph_vals.append(alph)
            errors.append(error)

    import pandas as pd
    df = pd.DataFrame({'alpha': alpha_vals, 'alph': alph_vals, 'errors': errors})

    import seaborn as sns
    plt.title('RMSE for Model 1, noise_var = 1')
    sns.barplot(x='alph', y='errors', hue='alpha', data=df)
    plt.ylim(0, 10)
    plt.savefig('rmse_model1.png', dpi=200)
    plt.show()

    #plt.bar(range(len(errors)), errors, tick_label=[str(x) for x in zip(alpha_vals, alph_vals)])

    import pickle
    with open('model_1_results.pkl', 'wb') as f:
        pickle.dump(rmse_vals, f)
    # plot pairs of profiles
    plt.figure(figsize=(10, 7))
    plt.title('A')
    plt.grid()
    plt.plot(results['time'], results['[A]'], label='original')
    plt.plot(results['time'], data[:,0], '--', label='noise')
    plt.plot(results_2['time'], results_2['[A]'], label='model')
    plt.legend()
    plt.savefig('plot_1.png', dpi=100)

    plt.figure(figsize=(10, 7))
    plt.title('B')
    plt.grid()
    plt.plot(results['time'], results['[B]'], label='original')
    plt.plot(results['time'], data[:,1], '--', label='noise')
    plt.plot(results['time'], results_2['[B]'], label='model')
    plt.legend()
    plt.savefig('plot_2.png', dpi=100)

    plt.figure(figsize=(10, 7))
    plt.title('C')
    plt.grid()
    plt.plot(results['time'], results['[C]'], label='original')
    plt.plot(results['time'], data[:,2], '--', label='noise')
    plt.plot(results['time'], results_2['[C]'], label='model')
    plt.legend()
    plt.savefig('plot_3.png', dpi=100)
    #basis_vectors, all_combs = basis_functions.generate_basis_functions(data, d=1)
    #derivs = basis_functions.calculate_derivatives(data, regularize=True, time_steps=1, alph=1e-1)
    #plt.plot(derivs)
    #plt.show()
    ###################################################################################

    data_mrna = pd.read_csv('Scipio_Wild_0_RNASeq.csv')
    data_protein = pd.read_csv('Scipio_Wild_0_MassSpec.csv')
    data_mrna['input'] = [1]*data_mrna.shape[0]
    all_vars = data_mrna.columns[1:].append(data_protein.columns[1:])
    print(all_vars)
    data_array = data_mrna.iloc[::2, 1:].values
    data_p_array = data_protein.iloc[:, 1:].values
    data_array = np.hstack([data_array, data_p_array])
    time_step = data_protein.time[1] - data_protein.time[0]
    eq = run_sindy(data_array, all_vars, d=3, time_steps=time_step, alph=10, alpha=1e-2, fit_intercept=True, plot=True)
    model2 = te.loada(eq)
    results_2 = model2.simulate(0, data_mrna.time.iloc[-1], data_array.shape[0])
    data_2 = np.column_stack([results_2['[' + x + ']'] for x in all_vars])
    error = np.sqrt(np.mean((data_2 - data_array)**2))
    print(error)

    plt.title('P1')
    plt.plot(data_protein['time'], data_protein['P1'], label='data')
    plt.plot(data_protein['time'], results_2['[P1]'], label='model')
    plt.show()

    plt.title('P6')
    plt.plot(data_protein['time'], data_protein['P6'], label='data')
    plt.plot(data_protein['time'], results_2['[P6]'], label='model')
    plt.show()
