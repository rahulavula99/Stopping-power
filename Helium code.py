# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:37:27 2021

@author: tiajo
"""
#Attempt to write a code to do a numerical integration technique, Simpson's rule

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Helium.csv', delimiter = ',')
def calculate_spacing(E_1, E_2):
    h = (E_1- E_2)/2
    return h

def simpson_rule(f_0, f_1, f_2, h):
    integral = h*(f_0 + 4*f_1 +f_2)/3
    return integral

def Bethe_bloch(E_2, c):
    return 1/((2.043*(10**26))*(1/(E_2))*(np.log(E_2)+6.307-(np.log(c)))*(10**-19))

data = np.genfromtxt('He.csv', delimiter = ',')


y_data=data[:,1] * (10**3) * 1.6 * (10**-19) 
x_data=data[:,0] * (10**-3)
channel=((data[:,1]+32)/0.72)*(10**3)*1.6*(10**-19)
uncertainty=data[:,2] * (10**3) * 1.6 * (10**-19) 
coeff = np.polyfit(x_data, y_data, 2)
yeet=(coeff[2]*(10**-3))/(1.6*(10**-19))

scale=(4664.128652/yeet)
print(scale)

#Energy against thickness graph
E=y_data*scale
unc_E=np.sqrt(((((scale*channel)**2)*(1.76*(10**-2))**2))+(((scale)**2)*((5.8*(10**3)*(1.6*10**-19))**2)))
coeff=np.polyfit(x_data,E,2)
fitfunction = np.poly1d(coeff)
r=np.linspace(min(x_data), max(x_data), 1000000)
raw_data_figure = plt.figure()
raw_data_plot = raw_data_figure.add_subplot(111)
raw_data_plot.set_title('Energy(J) against thickness(m) for Helium', fontsize=12)
raw_data_plot.set_xlabel('Thickness(m)')
raw_data_plot.set_ylabel('Energy(J)')
plt.plot(r, fitfunction(r))
raw_data_plot.errorbar(x_data,E,yerr=unc_E,fmt = 'o')
plt.grid(which = 'major', color='grey', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='black', linestyle = ':', linewidth=0.5)
plt.savefig("E_thicknessAr")
plt.show()
chi_square=np.sum((E-fitfunction(x_data))**2/((unc_E)**2))
print(chi_square/(len(x_data)-3))

#de/dx against E

x=np.linspace(min(E), max(E), 10000)
de = -((2*(coeff[0])*x_data)+coeff[1])
error_de=np.sqrt((2*coeff[0])**2*(data[:,3]*(10**-3))**2)
coeff2 = np.polyfit(E, de, 2)
fitfunction2=np.poly1d(coeff2)
raw_data_figure = plt.figure()
raw_data_plot = raw_data_figure.add_subplot(111)
raw_data_plot.set_title('-dE/dx (J/m) against Energy(J) for Nickel',fontsize=12)
raw_data_plot.set_xlabel('Energy(J)')
raw_data_plot.set_ylabel('-dE/dx(J/m)')
plt.plot(x, fitfunction2(x))
raw_data_plot.errorbar(E, de, yerr=error_de, fmt = 'o')
plt.savefig("de_Ni")
plt.show()
chi_square=np.sum(((de-fitfunction2(E))/(error_de))**2)
print(chi_square/(len(x_data)-3))

#1/(-dE/dx) against E

SI=(10**3) * 1.6 * (10**-19) 
initial_E=1/fitfunction2(4664.128652*SI)
final_E=1/fitfunction2(E)
h=calculate_spacing(4664.128652*SI,E)
mid=E+h
mid_E=1/fitfunction2(mid)
integral= simpson_rule(final_E, mid_E, initial_E,h)
print(integral)
unc_E2=np.sqrt(((((scale*SI*((4664.128652/scale)+32)/0.72)**2)*(1.76*(10**-2))**2))+(((scale)**2)*((5.8*(10**3)*(1.6*10**-19))**2)))

unc_initial_E=-(unc_E2)*(((2*coeff2[0]*(4664.128652*SI))+coeff2[1])/(((coeff2[0]*(4664.128652*SI)**2)+(coeff2[1]*4664.128652*SI)+coeff2[2])**2))
unc_final_E=-(unc_E)*(((2*coeff2[0]*(E))+coeff2[1])/(((coeff2[0]*(E)**2)+(coeff2[1]*E)+coeff2[2])**2))
unc_h=1/2*((unc_E2**2)+(unc_E**2))**(1/2)
unc_mid=(((unc_E)**2)+((unc_h)**2))**(1/2)

unc_mid_E=-(unc_mid)*(((2*coeff2[0]*(mid))+coeff2[1])/(((coeff2[0]*(mid)**2)+(coeff2[1]*mid)+coeff2[2])**2))
unc_integral=((((initial_E+4*mid_E+initial_E)*unc_h/3)**2 + (h*unc_initial_E/3)**2 + (4*h*unc_mid_E/3)**2 + (h*unc_final_E/3)**2)**(1/2))
print(unc_integral)
print(integral)
#Bethe-bloch formula

M_E=data[:,1]*scale*(10**-3)
M_final=4664.128652*(10**-3)
h_2=calculate_spacing(M_final,M_E)
M_mid=M_E+h
ion_E=np.linspace(52.73, 94.73, 10000)
chi=[]
for i in ion_E:
    f_1=Bethe_bloch(M_E, i)
    f_2=Bethe_bloch(M_mid, i)
    f_3=Bethe_bloch(M_final, i)
    integral2 = simpson_rule(f_1, f_2, f_3,h_2) *10**6
    chi=np.append(chi, np.sum(((integral2-x_data)/unc_integral)**2))

plt.figure()
plt.title('Chi^2 against ionization energy(eV)', fontsize=14)
plt.xlabel('Ionization energy(eV)', fontsize=14)
plt.ylabel('Chi^2', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(ion_E,chi)
ion_min=(ion_E[np.argmin(chi)])

MINIMUM_CHI_SQUARED = np.min(chi)
SIGMA_INDEX = np.argmin(np.abs(chi - MINIMUM_CHI_SQUARED - 1))
plt.scatter(ion_min, MINIMUM_CHI_SQUARED, s=100, label='minimum',
            c='k')
plt.legend(fontsize=14)
plt.show()


plt.plot()
plt.plot(ion_E, np.full(len(ion_E),
                                     MINIMUM_CHI_SQUARED + 1), c='grey',
         dashes=[1, 1], label=r'1 \sigma')


plt.legend(fontsize=14)
plt.savefig("Chi_He")
SIGMA = np.abs(ion_E[SIGMA_INDEX] - ion_min)

print('We find C = {0:4.2f} +/- {1:4.2f} with a reduced chi square of'
      ' {2}.'.format(ion_min, SIGMA,
                          MINIMUM_CHI_SQUARED / (len(x_data) - 3)))

"""
raw_data_figure = plt.figure()
raw_data_plot = raw_data_figure.add_subplot(111)
raw_data_plot.set_title('Differential range against Energy')
raw_data_plot.set_xlabel('Energy(J)')
raw_data_plot.set_ylabel(' \u0394R(m)')
raw_data_plot.errorbar(E_2, integral2, fmt = 'o')
plt.show()
"""
"""
#reduced chi-sqaured graph
c=np.linspace(25, 40, 10000)
chi=[]
for i in c:
    f_r1=range_graph2(E_2I, i)
    f_r2=range_graph2(x_1I, i)
    f_r3=range_graph2(E_1I, i)
    integral2 = simpson_rule(f_r1, f_r2, f_r3,h_2) *10**6
    chi=np.append(chi, np.sum((integral-integral2)/(0.1*(10**-6)))**2)

plt.figure()
plt.title('Chi^2 against coeffienct values', fontsize=14)
plt.xlabel('Coefficient values', fontsize=14)
plt.ylabel('Chi^2', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(c, chi)
"""

"""
popt,pcov = curve_fit(range_graph, y_data, 1/de )

raw_data_figure = plt.figure()
raw_data_plot = raw_data_figure.add_subplot(111)
raw_data_plot.set_title('Differential range against Energy')
raw_data_plot.set_xlabel('Energy(J)')
raw_data_plot.set_ylabel(' \u0394R(m)')
plt.plot(x, range_graph(x, *popt))
raw_data_plot.errorbar(E_2, integral, fmt = 'o')
plt.show()



#Finding the ionisation energy from minimising the chi-squared
"""
"""
c=np.linspace(9*10**-11, 9.5*10**-11, 1000)
def chi_squared(c, y_data):
    prediction = range_graph(y_data, popt[0], c)
    return np.sum(((-integral-prediction)/(0.5*(10**-6)))**2)
chi_square=[]
for coefficient in c:

    chi_square = np.append(chi_square, chi_squared(coefficient, y_data))


plt.figure()
plt.title('Chi^2 against coeffienct values', fontsize=14)
plt.xlabel('Coefficient values', fontsize=14)
plt.ylabel('Chi^2', fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.plot(c, chi_square)
"""



    


