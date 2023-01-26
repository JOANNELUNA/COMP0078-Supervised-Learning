
# ## PART II Supervised learning Coursework K-NN section

import numpy as np
import matplotlib.pyplot as plt



#==========================================================================================
#==========================================================================================

# defining a function to generate the set of pairs S
def h_sampling(N):

    # each x is uniformaly distributed over [0,1]^2
    x = np.random.uniform(size = 2*N).reshape(N,2)
    # y is randomly generated between 0 or 1
    y = np.random.randint(0,2, size = N)
    
    return x,y

X, y = h_sampling(100)


def K_NN (X_data, y_data, t_data, K):
    predict = []
    for x_t in t_data:

        # calculating the distance using L2 norm
        distance = np.linalg.norm(X_data- x_t, ord = 2, axis = 1)

        # sorting data by its distance from t
        nearest_neighbor = y_data[np.argsort(distance)]

        # since y is either 0 or 1, we can know which class is majority by
        # calculating the mean 
        y_t = np.mean(nearest_neighbor[:K])

        if y_t == 0.5:

            # undefined result, classify the data randomly
            result = np.random.randint(0,2)
        elif y_t > 0.5:
            result = 1
        else:
            result = 0
        predict.append(result)
    
    return np.array(predict)


#==========================================================================================
#==========================================================================================

# ## Question 6


# plotting decision boundary aand visualise K-NN
def K_NN_plot(X_data, y_data, K):

    #setting the grid of points
    x_grid, y_grid = np.meshgrid(np.arange(0,1.1, 0.01),np.arange(0,1.1, 0.01))
    n,d = x_grid.shape

    # formating and generate predictions
    pred = K_NN(X_data, y_data, np.c_[x_grid.ravel(), y_grid.ravel()], K).reshape(n,d)
    
    
    plt.figure(figsize = (8,5))

    # plotting decision boudary by using countourf function
    plt.contourf(x_grid, y_grid, pred, cmap='Blues')

    plt.scatter(X_data[:, 0], X_data[:, 1], c = y, cmap = 'Blues')

    plt.title('Sample data', fontsize  = 17)


K_NN_plot(X, y, 3)


#==========================================================================================
#==========================================================================================

# ## Question 7


# generate ph dataset from h:
def ph_sampling(X_data, y_data, N, K):
    
    sample_y = []
    sample_X = np.random.uniform(size = N*2).reshape(N, 2)
    
    # y is generated differently via flipping biased coin
    for i in range(N):

        cflip = np.random.choice([0,1], p =[0.2, 0.8])
        if cflip == 0:
            # tail, y sampled randomly
            sample_y.append(np.random.randint(0,2))
        else:
            # head, y is generated from hsv(x)
            sample_y.append(K_NN(X_data,y_data, sample_X[i,:].reshape(-1,2), K).item())
    
    return sample_X, np.array(sample_y)



def protocal_A(num_train, num_test, K):
    gen_error = []
    for k in range(1, 50):
        iter_error = 0
        for i in range(100):
            # generate a h-distribution
            X_data, y_data = h_sampling(100) 

            # generate our training set of 4000 data
            X_train, y_train = ph_sampling(X_data, y_data,num_train, K)

            # generate our test set of 1000 data
            X_test, y_test = ph_sampling(X_data, y_data, num_test, K)

            # generate prediction for test set
            pred = K_NN(X_train, y_train, X_test, k)

            # calculating generalised error for each iteration
            iter_error += np.mean((pred != y_test))

        # average geeneralised error
        gen_error.append(iter_error/100)

    return gen_error



# generating averaged generalised error for k in (1,50):
error = protocal_A(4000, 1000, 3)

# plotting the result
plt.figure(figsize=(8,5))

plt.plot(range(1,50),list(error))
plt.title('generalised error against k', fontsize = 17)
plt.xlabel('k-value', fontsize  = 14)
plt.ylabel('average generalised error', fontsize  = 14)


#==========================================================================================
#==========================================================================================

# ## Question 8

def protocol_B(K):
    k_for_each_m = []
    for m in (100, 500,1000,1500,2000,2500,3000,3500,4000):
        optimal_k = 0
        for i in range(100):
            k_gen_error = []
            for k in range(1,50):

                # similar procedure as protocal A
                X_data, y_data = h_sampling(100)

                X_train, y_train = ph_sampling(X_data, y_data,m, K)

                X_test, y_test = ph_sampling(X_data, y_data, 1000, K)

                pred = K_NN(X_train, y_train, X_test, k)

                k_gen_error.append(np.mean(pred != y_test))
            
            # determining and summing up the optimal k for each iteration
            optimal_k += (np.argmin(k_gen_error)+1)
        
        # calculating and storing the average of optimal k over the 100 iters 
        # for each training size
        k_for_each_m.append(optimal_k/100)

    return k_for_each_m


ave_optimal_k = protocol_B(3)

plt.figure(figsize=(8,5))
plt.plot((100, 500,1000,1500,2000,2500,3000,3500,4000), ave_optimal_k)
plt.title('average optimal k against training size',fontsize  = 17)
plt.ylabel('average optimal k-value',fontsize  = 14)
plt.xlabel('training size',fontsize  = 14)