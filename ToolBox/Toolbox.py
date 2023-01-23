import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from Toolbox import *
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import pacf
import seaborn as sns


def cal_rolling_mean_var(x,y):
    rmean = [x[0]]
    rvariance =[]
    for k in range(1, len(x)):
        result_mean = np.mean(x[:k])
        rmean.append(result_mean)
    for j in range(1,len(x)+1):
        result_variance = np.var(x[:j])
        rvariance.append(result_variance)
    print(f"The rolling mean of is :", rmean[:10])
    print(f"The rolling variance of is :", rvariance[:10])

##To plot the rolling mean :
    plt.plot(y, rmean , color='Red')
    plt.ylabel(f'Pollution')
    plt.xlabel(f'Time(Daily)')
    plt.title(f"The Rolling mean of pollution(pm2.5) concentration")
    plt.figure()
    plt.show()
##To plot the rolling variance:
    plt.plot(y , rvariance , color='Yellow')
    plt.ylabel(f'Pollution')
    plt.xlabel(f'Time(Daily)')
    plt.title(f"The Rolling variance of pollution(pm2.5) concentration")
    plt.figure()
    plt.show()

from statsmodels.tsa.stattools import adfuller
def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
     print('\t%s: %.3f' % (key, value))



from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries):
     print(f'Results of KPSS Test for {timeseries}:')
     kpsstest = kpss(timeseries, regression='c', nlags="auto")
     kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags-Used'])
     for key, value in kpsstest[3].items():
         kpss_output['Critical Value (%s)' % key] = value
     print(kpss_output)


def first_order_differencing(dataset, interval=1):
    difference = []
    difference[0]=difference.append(np.nan)
    for i in range(interval , len(dataset)):
	    value = dataset[i]-dataset[i - interval]
	    difference.append(value)
    return Series(difference)

def second_order_differencing(dataset ,interval=2):
    difference = []
    difference[0]=difference.append(np.nan)
    difference[1]=difference.append(np.nan)
    for i in range(interval , len(dataset)+1):
        value = dataset[i]-dataset[i - 1]
        difference.append(value)
    return Series(difference)

def third_order(dataset , interval=3):
    difference = []
    difference[0]=difference.append(np.nan)
    difference[1]=difference.append(np.nan)
    difference[2]=difference.append(np.nan)
    for i in range(interval ,len(dataset)+2):
        value = dataset[i]-dataset[i-1]
        difference.append(value)
    return Series(difference)
import math
def correlation_coefficient_cal(data1,data2):
    n = len(data1)
    #for calculating the mean:
    data1_mean = np.mean(data1)
    data2_mean = np.mean(data2)
    numerator = 0
    for i, j in zip(data1,data2):
        numerator+= (i-data1_mean) * (j-data2_mean)
    sum1=0
    sum2=0
    for i in data1:
        sum1+=(i-data1_mean)**2
    sqrt_data1 = np.sqrt(sum1)
    for j in data2:
         sum2+=(j-data2_mean)**2
    sqrt_data2 = np.sqrt(sum2)
    denominator=sqrt_data1*sqrt_data2
    rho = numerator/denominator
    return (rho)

def auto_correlation(y):
    lags = int(input("Enter the lags:"))
    T = len(y)
    numerator = 0
    denominator = 0
    list_acf = []
    y_mean = np.mean(y)
    d_t=0
    for tho in range(d_t, T):
        denominator += (y[tho] - y_mean) ** 2
    print(denominator)
    for i in range(0 , lags):
        for t in range(i , T):
            numerator+= (y[t]-y_mean)*(y[t-i]-y_mean)
        acf = numerator/denominator
        numerator=0
        list_acf.append(acf)

    print("ACF is :",list_acf)
    return list_acf


import numpy as np
# define moving average function
def moving_avg(x):
    n = int(input("Enter the n value :"))
    if (n % 2) != 0:
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)
    elif (n % 2) == 0:
        m= 2
        cumsum = np.cumsum(np.insert(x, 0, 0))
        result_four = (cumsum[n:] - cumsum[:-n]) / float(n)
        cumsum_kfold = np.cumsum(np.insert(result_four, 0, 0))
        result_2ma = (cumsum_kfold[m:] - cumsum_kfold[:-m]) / float(m)
        return result_2ma



##ARMA process :
def ARMA():
    samples = int(input("Enter the samples :"))
    mean_wn = float(input("Enter the mean of white noise :"))
    variance = float(input("Enter the variance:"))
    na = int(input("Enter the ar order:"))
    nb = int(input("Enter the ma order :"))
    an = [0]*na
    bn = [0]*nb
    for i in range(na):
        an[i] = float(input("Enter the co-efficient :"))

    for j in range(nb):
        bn[j] = float(input("Enter the co-efficient :"))

    max_order = max(na , nb)
    num = [0]*(max_order+1)
    den = [0]*(max_order+1)
    for i in range(na+1):
        if i ==0:
            den[i]=1
        else:
            den[i]=an[i-1]

    arparams = np.array(an)
    print(arparams)
    maparams = np.array(bn)
    print(maparams)
    na=len(arparams)
    nb=len(maparams)
    ar = np.r_[1 , arparams]
    ma = np.r_[1 , maparams]
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    if mean_wn == 0:
       y = arma_process.generate_sample(samples)
    else:
       mean_y = mean_wn * (1 + np.sum(bn)) / (1 + np.sum(an))
       y = arma_process.generate_sample(samples, scale=np.sqrt(variance) + mean_y)
    return y


##GPAC process:
def calc_Gpac(na_order, nb_order, Ry):
    x = int((len(Ry) - 1) / 2)
    df = pd.DataFrame(np.zeros((na_order, nb_order + 1)))
    df = df.drop(0, axis=1)

    for k in df:  # this for loop iterates over the column to calculate the value
        for j, row_val in df.iterrows():  # this iterates over the rows
            if k == 1:  # for first column
                dinom_val = Ry[x + j]  # Here Ry(0) = lags -1   = x
                numer_val = Ry[x + j + k]
            else:  # when our column is 2 or more than 2; when k > 2
                dinom_matrix = []
                for rows in range(k):  # this loop is for calculating the square matrix (iterating over the rows of matrix)
                    # print(rows)
                    row_list = []
                    for col in range(k):  # this loop is for calculating the square matrix (iterating over the columns of matrix)
                        # print(col)
                        each = Ry[x - col + rows + j]
                        # print(each)
                        row_list.append(each)
                    dinom_matrix.append(np.array(row_list))

                # dinominator matrix and numerator matrix have same values except for the last column so:
                dinomator_matrix = np.array(dinom_matrix)
                numerator_matrix = np.array(dinom_matrix)

                # updating values for last column of numerator matrix

                last_col =k
                for r in range(k):
                    numerator_matrix[r][last_col - 1] = Ry[x + r + 1 + j]

                # calculating determinants
                numer_val = np.linalg.det(numerator_matrix)
                dinom_val = np.linalg.det(dinomator_matrix)

            df[k][j] = numer_val / dinom_val  # plugs the value in GPAC table

    print(df)

    import seaborn as sns
    sns.heatmap(df, cmap=sns.diverging_palette(20, 220, n=200), annot=True, center=0)
    plt.title('Generalized Partial Auto-correlation Table')
    plt.xlabel("K-values")
    plt.ylabel("J-values")
    plt.show()


from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('Auto Correlation Plot')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.subplot(212)
    plt.title("Partial Auto Correlation Plot")
    plot_pacf(y, ax=plt.gca(), lags=lags)
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    fig.tight_layout(pad=3)
    plt.show()


from scipy import signal

def calc_theta(y,na,nb, theta):
    if na == 0:
        dinominator = [1]
    else:
        dinominator =np.append([1], theta[:na])

    if nb ==0:
        numerator = [1]
    else:
        numerator = np.append([1], theta[-nb:])

    diff = na -nb
    if diff > 0:
        numerator =np.append(numerator, np.zeros(diff))


    sys = (dinominator, numerator, 1)
    _,e = signal.dlsim(sys, y)
    theta =[]
    for i in e:
        theta.append(i[0])

    theta_e =np.array(theta)

    return theta_e

def step_1(y, na, nb, theta, delta):
    e = calc_theta(y,na,nb, theta)
    SSE = np.dot(e, e.T)

    X=[]

    n =na +nb

    for i in range(n):
        theta_new =theta.copy()
        theta_new[i] =theta_new[i] +delta
        new_e = calc_theta(y, na, nb, theta_new)
        x_i = np.subtract(e, new_e)/delta
        X.append(x_i)

    X =  np.transpose(X)
    A =  np.transpose(X).dot(X)
    g=  np.transpose(X).dot(e)

    return A,g, SSE

def step_2(theta, A, g, u, y,na,nb):
    n = na +nb
    idt = np.identity(n)
    before_inv= A + (u * idt)
    AUI_inv = np.linalg.inv(before_inv)
    diff_theta= AUI_inv.dot(g)
    theta_new = theta +diff_theta

    new_e = calc_theta(y, na, nb, theta_new)
    SSE_new = new_e.dot(new_e.T)
    return SSE_new, theta_new, diff_theta

def calc_LMA(count, y, na, nb, theta, delta, u, u_max):
    i =0
    SSE_count=[]
    norm_theta=[]

    while i <count:
        A,g,SSE = step_1(y,na,nb,theta, delta)
        SSE_new, theta_new, diff_theta =step_2(theta, A, g,u, y,na,nb)
        SSE_count.append(SSE_new)

        n =na+nb

        if SSE_new < SSE:
            norm_theta2  = np.linalg.norm(np.array(diff_theta),2)
            norm_theta.append(norm_theta2)

            if norm_theta2 < 0.001:
                theta = theta_new.copy()
                break

            else:
                theta =theta_new.copy()
                u = u/10

        while SSE_new >= SSE:
            u = u *10
            if u>u_max:
                print("Mue is high now and cannot go higher than that!!!")
                break
            SSE_new, theta_new, diff_theta = step_2(theta, A, g, u, y,na,nb)
        theta  = theta_new
        i += 1

    variance_error = SSE_new / (len(y) - n)
    co_variance =variance_error * np.linalg.inv(A)
    print("The estimated parameters >>> ", theta)
    print(f"\n The estimated co-variance matrix is {co_variance}")
    print(f"\n The estimated variance of error is {variance_error}")


    for i in range(na):
        std_deviation = np.sqrt(co_variance[i][i])
        print(f"The standard deviation for a{i+1} is {std_deviation}")

    for j in range(na, n):
        std_deviation = np.sqrt(co_variance[j][j])
        print(f"The standard deviation for b{i + 1} is {std_deviation}")

    print(f"\n The confidence interval for parameters are: ")
    for i in range(na):
        interval = 2 * np.sqrt(co_variance[i][i])
        print(f"{(theta[i]- interval)} < a{i+1} < {(theta[i] + interval)}")

    for j in range(na,n):
        interval = 2 * np.sqrt(co_variance[j][j])
        print(f"{(theta[j]- interval)} < b{j -na + 1} < {(theta[j] + interval)}")


    #zero/pole

    num_root =[1]
    den_root= [1]

    for i in range(na):
        num_root.append(theta[i])
    for i in range(nb):
        den_root.append(theta[i + na])

    poles =np.roots(num_root)
    zeros = np.roots(den_root)

    print(f"\nThe roots of the numerators are {zeros}")
    print(f"The roots of dinominators are {poles}")

    #plt.plot(SSE_count)
    #plt.xlabel("Numbers of Iternations")
    #plt.ylabel("Sum square Error")
    #plt.title("Sum of square Error vs the iterations")
    #plt.show()

    return theta,co_variance, SSE_count

def SARIMA_func():
    T=int(input('Enter number of samples'))
    mean=eval(input('Enter mean of white nosie'))
    var=eval(input('Enter variance of white noise'))
    na = int(input("Enter AR process order"))
    nb = int(input("Enter MA process order"))
    naparam = [0] * na
    nbparam = [0] * nb
    for i in range(0, na):
        naparam[i] = float(input(f"Enter the coefficient of AR:a{i + 1}"))
    for i in range(0, nb):
        nbparam[i] = float(input(f"Enter the coefficient of MA:b{i + 1}"))
    while len(naparam) < len(nbparam):
        naparam.append(0)
    while len(nbparam) < len(naparam):
        nbparam.append(0)
    ar = np.r_[1, naparam]
    ma = np.r_[1, nbparam]
    e=np.random.normal(mean,np.sqrt(var),T)
    system=(ma,ar,1)
    t,process=signal.dlsim(system,e)
    a=[a[0] for a in process]
    return a


def difference(y,interval):
    diff=[]
    for i in range(interval,len(y)):
        value=y[i]-y[i-interval]
        diff.append(value)
    return diff




