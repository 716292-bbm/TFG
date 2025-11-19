import uproot
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import norm

file = uproot.open("Datos/BEhistos_year123456.root")
file.keys()

def calcula_t_exposicion(array_det=np.ones(9,dtype=int)):
#  Calculamos el tiempo total de exposicion

    exposiciones=np.zeros(9) #dias
    
    exposiciones[0]=2031.38
    exposiciones[1]=2033.20
    exposiciones[2]=2029.52
    exposiciones[3]=2022.55
    exposiciones[4]=2033.01
    exposiciones[5]=2030.18
    exposiciones[6]=2032.27 
    exposiciones[7]=2031.02
    exposiciones[8]=2020.29

    dias=0
    for i in range(9):
        if(array_det[i]==1):
            dias+=exposiciones[i]
    return dias

def calcula_m_exposicion(array_det=np.ones(9,dtype=int)):
#  Calculamos el tiempo total de exposicion
    masa_detector=0
    for i in range(9):
            if(array_det[i]==1):
                masa_detector+=12.5
    return masa_detector

#################
# Funcion inversa de la CDF de la distribucion gaussiana. (Devuelve el valor del limite superior de la integral gaussiana desde 
# menos infinito que da como resultado un area p (en tanto por uno),devuelve el valor en multiplos de sigma)
#################
# p: area de la gausiana para la cual calculamos el limite superior
def f_significancia(p):
    return norm.ppf(p)

#################
# Calcula el ritmo experimental
#################
# E_i: Limite inferior de la energia (keV_ee) (Minimo 1 keV_ee)
# E_f: Limite superior de la energia (keV_ee) (Maximo 30 keV_ee)
# signif: Intervalo de confianza de la medida

def integral_exp(E_i,E_f,masa_det,dias,signif=0.9, array_det=np.ones(9,dtype=int)):

    if(E_i<1): 
        print('El valor de E_i debe ser mayor que 1 keV_ee')
        return "El valor de E_i debe ser mayor que 1 keV_ee"
    if(E_f>30): 
        print('El valor de E_f debe ser menor que 30 keV_ee')
        return "El valor de E_f debe ser menor que 30 keV_ee"

    es=np.arange(1,30,0.1)
    datos=np.zeros((19,len(es)))

    #   Cargamos los datos del histograma del archivo root (en c/kg/keV_ee/dia) 

    for i in range (9):
        datos[2*i+1]=file["hbea_123456y_D"+str(i)].values()
        datos[2*i+2]=file["hbea_123456y_D"+str(i)].errors()
    datos[0]=es

    #   Hacemos la integral en el intervalo de energias definido por E_i y E_f

    ritmos_exp=np.zeros(9)
    errores_exp=np.zeros(9)

    for i in range (9):
        ritmos_exp[i]=0
        errores_exp[i]=0
        if(array_det[i]==1):
            for j in range(int(10*(E_i-1)),int(10*(E_f-1))):
                ritmos_exp[i]+=datos[2*i+1][j]
            ritmos_exp[i]=ritmos_exp[i]*0.1*12.5*

    ritmo_exp=0
    error_exp=0

    #   Sumamos el ritmo de todos los detectores (en c/kg/dia) 

    for i in range (9):
        if(array_det[i]==1):
            ritmo_exp+=ritmos_exp[i]
            error_exp+=(errores_exp[i]*errores_exp[i])
    error_exp=np.sqrt(error_exp)
    
    
    

    # Calculamos el ritmo final del detector (cuentas)

    ritmo_exp_final=ritmo_exp*dias*masa_det
    error_exp_final=error_exp*dias*masa_det

    # Aplicamos el factor de significancia

    delta=f_significancia(signif)*np.sqrt(ritmo_exp_final)
    ritmo_comp=ritmo_exp_final+delta
    return ritmo_comp
  