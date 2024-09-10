import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Solución de la ODE
def V(t):
  return E + (V0 - E) * np.exp(-t / tau)  # comportamiento de una neurona

# 2.2) Grafique la solución para  V0 = 10mV  y  t∈[0,100]ms.
# Incorpore al gráfico una línea punteada indicando el potencial de reposo.

E = -65  # mV
tau = 10  # ms
V0 = 10  # mV
valores_t = np.linspace(0, 100, 100)  # ms
valores_V = np.vectorize(V)(valores_t)

fig = go.Figure()
fig.add_trace(go.Scatter(x=valores_t, y=valores_V, mode='lines', name='Potencial de membrana', line=dict(color='red')))
fig.add_trace(go.Scatter(x=valores_t, y=np.zeros(len(valores_t)), mode='lines', name='Línea base', line=dict(color='gray')))
fig.add_trace(go.Scatter(x=valores_t, y=E*np.ones(len(valores_t)), mode='lines', name='$E = V^*$', line=dict(dash='dash', color='black')))
fig.update_layout(
    title='Integrate and Fire: sin disparo ni corriente',
    xaxis_title='Tiempo [ms]',
    yaxis_title='Potencial de membrana [mV]',
    legend_title='Leyenda'
)
fig.show()

# 2.3) Realice un análisis geométrico de la solución calculada. Incorpore flechas del campo vectorial
# así como los puntos fijos estables, inestables y marginales, si los hubiere.

Vfix = E  # punto fijo es el punto donde la derivada cruza por cero.
def f(V):
    return (E-V)/tau  # ecuación de una recta con pendiente negativa, con i(t) = 0
valores_V = np.linspace(-120,20,100)
valores_f = np.vectorize(f)(valores_V)
fig = go.Figure()
fig.add_trace(go.Scatter(x=valores_V, y=valores_f, mode='lines', line=dict(color='red'), name='')) # Línea roja
fig.add_trace(go.Scatter(x=valores_V, y=np.zeros(len(valores_V)), mode='lines', line=dict(color='gray', dash='dash'), name='')) # Línea gris
fig.add_trace(go.Scatter(x=[Vfix, Vfix], y=[-8, 8], mode='lines', line=dict(color='cyan', dash='dash'), name='$V^*=E$')) # Línea cian
# Flechas verdes
fig.add_annotation(x=Vfix-20, y=0, ax=Vfix-30, ay=0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green')
fig.add_annotation(x=Vfix+20, y=0, ax=Vfix+30, ay=0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='green')
fig.add_trace(go.Scatter(x=[Vfix], y=[0], mode='markers', marker=dict(color='black'), name='')) # Punto negro
# Títulos y etiquetas
fig.update_layout(
    title='Punto fijo en -65 mV (atractor)',
    xaxis_title='$V$ [mV]',
    yaxis_title='$dV/dt=f(V)$ [mV/ms]',
    showlegend=True
)
fig.show()

# 2.4) Repita 1), 2) y 3) para el caso de corriente de entrada constante, I=2nA.
# 2.4.2)   En este ejercicio, la corriente ya no es cero
E = -65 #mV
tau = 10 #ms
V0 = 10 #mV
I = 2 #nA
R = 10 #Mohms
Vfix = E+I*R
def V(t):
    return Vfix+(V0-Vfix)*np.exp(-t/tau)

plt.xlabel('$t$ [ms]')
plt.ylabel('$V$ [mV]')
valores_t = np.linspace(0,100,100)
valores_V = np.vectorize(V)(valores_t)
plt.plot(valores_t,valores_V, label = "", linestyle = '-', c = 'red')
plt.plot(valores_t,[Vfix]*len(valores_t), label = "$V^*$", linestyle = '--', c = 'cyan')
plt.plot(valores_t,np.zeros(len(valores_t)),label = "", linestyle = '--', c = 'gray')
plt.title('Integrate and Fire: sin disparo y corriente constante')
plt.legend()
plt.show()


# 2.4.3)
def f(V):
    return (Vfix-V)/tau

Vfix = E
def f(V):
    return (E-V)/tau
valores_V = np.linspace(-120,20,100)
valores_f = np.vectorize(f)(valores_V)

plt.xlabel('$V$ [mV]')
plt.ylabel('$dV/dt=f(V)$ [mV/ms]')
plt.plot(valores_V,valores_f, label = "", linestyle = '-', c = 'red')
plt.plot(valores_V,np.zeros(len(valores_V)),label = "", linestyle = '--', c = 'gray')
plt.plot([Vfix,Vfix],[-8,8],label="$V^*=E$", linestyle = '--', c = 'cyan')
plt.arrow(Vfix-20.0,0.0,10.0,0.0,head_width=0.5,head_length=2,fc='g',ec='g')
plt.arrow(Vfix+20.0,0.0,-10.0,0.0,head_width=0.5,head_length=2,fc='g',ec='g')
plt.scatter([Vfix],[0],c='black')
plt.title('Punto fijo (atractor)')
plt.legend()
plt.show()



