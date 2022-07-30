# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:50:42 2021

@author: rahul
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def calculate_spacing(E_1, E_2):
    h = (E_1- E_2)/2
    return h

def simpson_rule(f_0, f_1, f_2, h):
    integral = h*(f_0 + 4*f_1 +f_2)/3
    return integral
def Bethe_bloch(E_2, c):
    return 1/(9.727*(10**30)*(1/(E_2))*(np.log(E_2)+6.307-(np.log(c)))*(10**-19))
def quadratic(x,a,b,c):
    return a*x**2 + b*x +c 
data = np.genfromtxt('Ni.csv', delimiter = ',')
channel=((data[:,1]+18.7)/1.08)*(10**3)*1.6*(10**-19)
y_data=data[:,1] * (10**3) * 1.6 * (10**-19) 
x_data=data[:,0] * (10**-6)
uncertainty=data[:,2] * (10**3) * 1.6 * (10**-19) 
coeff = np.polyfit(x_data, y_data, 2)
yeet=(coeff[2]*(10**-3))/(1.6*(10**-19))

scale=(5796/yeet)
print(scale)
sqrt=(data[:,5])**(1/2)


#Energy against thickness graph

E=y_data*scale
unc_E=np.sqrt(((((scale*channel)**2)*(1.9*(10**-2))**2))+(((scale)**2)*((8.9*(10**3)*(1.6*10**-19))**2)))
t=x_data
popt, pcov = curve_fit(quadratic, t, E, sigma=unc_E)

r=np.linspace(min(x_data), max(x_data), 1000000)
raw_data_figure = plt.figure()
raw_data_plot = raw_data_figure.add_subplot(111)
raw_data_plot.set_title('Energy(J) against thickness(m) for Nickel', fontsize=12)
raw_data_plot.set_xlabel('Thickness(m)')
raw_data_plot.set_ylabel('Energy(J)')
plt.plot(r, quadratic(r,*popt))
raw_data_plot.errorbar(x_data,E,yerr=unc_E,fmt = 'o')
plt.grid(which = 'major', color='grey', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='black', linestyle = ':', linewidth=0.5)
plt.savefig("E_thicknessNi")
plt.show()
chi_square=np.sum((E-quadratic(t,*popt))**2/((unc_E)**2))
print(chi_square/(len(x_data)-3))
print(popt)


#de/dx against E


x=np.linspace(min(E), max(E), 10000)
de = -((2*(coeff[0])*x_data)+coeff[1])
error_de=np.sqrt((2*coeff[0])**2*(data[:,3]*(10**-6))**2)
coeff2 = np.polyfit(E, de, 2)
fitfunction2=np.poly1d(coeff2)
print(fitfunction2)
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
initial_E=1/fitfunction2(5796*SI)
final_E=1/fitfunction2(E)
h=calculate_spacing(5796*SI,E)
mid=E+h
mid_E=1/fitfunction2(mid)
integral= simpson_rule(final_E, mid_E, initial_E,h)
print(integral)
unc_E2=np.sqrt(((((scale*SI*(966.526+18.7)/1.08)**2)*(1.9*(10**-2))**2))+(((scale)**2)*((8.9*(10**3)*(1.6*10**-19))**2)))

unc_initial_E=-(unc_E2)*(((2*coeff2[0]*(5796*SI))+coeff2[1])/(((coeff2[0]*(5796*SI)**2)+(coeff2[1]*5796*SI)+coeff2[2])**2))
unc_final_E=-(unc_E)*(((2*coeff2[0]*(E))+coeff2[1])/(((coeff2[0]*(E)**2)+(coeff2[1]*E)+coeff2[2])**2))
unc_h=1/2*((unc_E2**2)+(unc_E**2))**(1/2)
unc_mid=(((unc_E)**2)+((unc_h)**2))**(1/2)

unc_mid_E=-(unc_mid)*(((2*coeff2[0]*(mid))+coeff2[1])/(((coeff2[0]*(mid)**2)+(coeff2[1]*mid)+coeff2[2])**2))
unc_integral=((((initial_E+4*mid_E+initial_E)*unc_h/3)**2 + (h*unc_initial_E/3)**2 + (4*h*unc_mid_E/3)**2 + (h*unc_final_E/3)**2)**(1/2))
print(unc_integral)
print(integral)
#Bethe-Bloch
M_E=data[:,1]*scale*(10**-3)
M_final=5796*(10**-3)
h_2=calculate_spacing(M_final,M_E)
M_mid=M_E+h
ion_E=np.linspace(300, 384.14, 10000)
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
                                     MINIMUM_CHI_SQUARED + 1), c='Red',
         dashes=[1, 1], label=r'1 sigma')
plt.grid(which = 'major', color='grey', linestyle='-', linewidth=0.5)
plt.grid(which = 'minor', color='black', linestyle = ':', linewidth=0.5)
plt.legend(fontsize=14)
plt.savefig("chi_Ni")

SIGMA = np.abs(ion_E[SIGMA_INDEX] - ion_min)

print('We find C = {0:4.2f} +/- {1:4.2f} with a reduced chi square of'
      ' {2}.'.format(ion_min, SIGMA,
                          MINIMUM_CHI_SQUARED / (len(x_data) - 3)))
