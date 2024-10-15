# Aprendizaje Federado CDMX
El **Aprendizaje Federado (FL)** es un enfoque de aprendizaje automático donde los modelos son entrenados en dispositivos distribuidos, sin que los datos de entrenamiento abandonen los dispositivos locales. Esta configuración permite la colaboración entre múltiples dispositivos o instituciones sin compartir datos sensibles. Este modelo fue introducido por Google en 2016 [@yang2019federated] y funciona en esencia en una estructura como en la Figura \ref{ArqFedLE}. 

En esta configuración, un servidor central inicializa los parámetros de un modelo que se quiere ajustar, luego cada cliente (aquellas personas o instituciones que cuentan con una base de datos) actualiza de manera local sus parámetros (parámetros locales) y devuelve estos al servidor central (sólo los parámetros, nunca los datos). Luego, este, habiendo recibido actualizaciones de todos los clientes, actualiza los parámetros de manera general (parámetros globales), y el ciclo se repite de nuevo hasta alcanzar una cantidad predeterminada de iteraciones. Esta configuración permite que las bases de datos que posea cada cliente nunca salgan de ellos, evitando así compartir información sensible. Esta idea queda planteada en el Algoritmo \ref{EsqGenFL}, en este la función `$ClientUpdate(k, w_t)$` representa la forma en que el cliente `$k$` actualizará sus parámetros a partir de los actuales. Más adelante se abordarán las posibles opciones.

![Arquitectura básica del Aprendizaje Federado. Extraído de Xu (2021)](IMG/ArqBas.PNG)

### Algoritmo: Esquema General FL

```algorithm
\caption{Esquema General FL} \label{EsqGenFL}
\begin{algorithmic}[1]
\State \textbf{Resultado:} Modelo global entrenado
\State \textbf{Inicializar:} $w_0$
\For{cada ronda global $t = 0, 1, \dots$}
    \For{cada cliente $k \in \{1, \dots, K\}$}
        \State $w_{t+1}^k \gets \text{ClientUpdate}(k, w_t)$
    \EndFor
    \State $w_{t+1} \gets \frac{1}{n} \sum_{k=1}^{K} n_k w_{t+1}^k$
\EndFor
\end{algorithmic}
\end{algorithm}
