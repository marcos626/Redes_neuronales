import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from ODE import euler, rk4, integrador_ode

# 3.1) Resuelva numéricamente la ODE del modelo Integrate and Fire en el intervalo $t\in [0,200]ms$
# para la condición inicial $V(0)=E$ y una corriente de entrada $I(t)=I_0$ para
# todo t con I_0=2nA. Utilice el paso de integración h=0.05ms.

E = -65  # mV
tau = 10  # ms tiempo de relajación
V0 = 10  # mV
Vu = -50  # mV tensión umbral o threshold
R = 10  # Mohms

def I(t):   #Tuve que agregar la definición de la función I, no está en la notebook del profe
  return 2  # nA
p = [E,tau,R,I,Vu]  # parámetros
def f(x,t,p):
  V   = x[0]
  E   = p[0]
  tau = p[1]
  R   = p[2]
  I   = p[3]  #I : t -> I(t)
  #Vu  = p[4]
  return np.array([(E+R*I(t)-V)/tau])
def c(x,t,p):  # (condición)
  V   = x[0]
  E   = p[0]
  #tau = p[1]
  #R   = p[2]
  #I   = p[3]  #I : t -> I(t)
  Vu  = p[4]
  if V>Vu:
      V=E
  return np.array([V])

a = 0  # ms
b = 200  # ms
xa = np.array([E])
h = 0.05  # ms
k = int((b-a)/h)
t,w = integrador_ode(rk4,f,xa,a,b,k,p,c=c)
w[0,:]


# 3.2) Grafique la solución computada en el inciso 3.1). Incorpore al gráfico la solución
# exacta del inciso **2.1)**, en donde el mecanismo de disparo está desactivado, para la
# misma condición inicial del inciso **3.1)**. Grafique, también, líneas punteadas horizontales
# marcando los valores de $V^*$ y $V_u$, donde $V^*$ es el valor de $V$ tal que $f(V)=0$.

V0 = E
Vfix = E+I(0)*R
def Vex(t):
    return Vfix+(V0-Vfix)*np.exp(-t/tau)


#3.3) Para el caso indicado en el inciso 3.1), calcule analíticamente el período de disparo.
#Observe si el valor computado corresponde con el observado en el inciso 3.2).
#Grafíque la frecuencia en función de I_0.
#Que ocurre para valores pequeños de I_0?

# corriente crítica
Ic = (Vu-E)/R
print(f"Corriente crítica: {Ic} nA")

def periodo(I0):
  return tau * np.log(I0 * R / (I0 * R + E - Vu))
def frecuencia(I0):
  return 1 / periodo(I0)

valores_I0 = np.linspace(Ic + 0.01, 2 * Ic, 10)
frecuencias = np.vectorize(frecuencia)(valores_I0)
valores_I0 = np.linspace(Ic + 0.01, 2 * Ic, 10)

plt.xlabel('$I_0$ [nA]')
plt.ylabel('Frecuencia [1/ms]')
plt.scatter(np.linspace(0, Ic, 10), np.zeros(10), label="", linestyle='--', c='blue')
plt.scatter(valores_I0, np.vectorize(frecuencia)(valores_I0), label="", linestyle='-', c='blue')
plt.title('Integrate and fire: frecuencia de disparo')
# plt.legend()
plt.show()


# Crear la figura con plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=valores_I0, y=frecuencias, mode='markers', name='Frecuencia', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=np.linspace(0, Ic, 10), y=np.zeros(10), mode='markers', name='Corriente Crítica', line=dict(color='blue', dash='dash')))
fig.update_layout(
    title='Integrate and fire: frecuencia de disparo',
    xaxis_title='$I_0$ [nA]',
    yaxis_title='Frecuencia [1/ms]',
    legend_title='Leyenda'
)
fig.show()



