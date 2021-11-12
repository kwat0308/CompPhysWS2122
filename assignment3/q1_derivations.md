From the sheet, we have the expression for the average magnetization per site as follows:

$
\begin{equation*}
    <m> = \dfrac{1}{N\beta}\dfrac{\partial}{\partial h}\left(\log Z \right) = \dfrac{1}{N\beta} \dfrac{1}{Z} \dfrac{\partial Z}{\partial h}
\end{equation*}
$

So using the partition function (with $J > 0$) given in the exercise sheet, we have that:

$\begin{align*}
    \dfrac{\partial Z}{\partial h} &= \dfrac{\partial}{\partial h} \left( \int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} \exp{\left[ -\frac{\phi^2}{2\beta\hat{J}} + N\log\left( 2 \cosh(\beta h \pm \phi)\right)\right]}\right) \\
    &=\int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} \dfrac{\partial}{\partial h} \left( \exp{\left[ -\frac{\phi^2}{2\beta\hat{J}} + N\log\left( 2 \cosh(\beta h \pm \phi)\right)\right]}\right) \\
    &:= \int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} \dfrac{\partial}{\partial h} \left( \exp{\left[\kappa(\beta, h)\right]}\right) \\
    &= \int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} e^{\kappa(\beta, \phi)} \dfrac{\partial \kappa}{\partial h}
\end{align*}$

Now:

$
\begin{equation*}
\dfrac{\partial \kappa}{\partial h} = N \dfrac{2 \sinh(\beta h \pm \phi)}{2 \cosh(\beta h \pm \phi)} \beta = N\beta \tanh(\beta h \pm \phi)
\end{equation*}$

Substituting Eq. (3) into Eq. (2), and thus substituting this into Eq. (1), we get our average magnetization per site as follows:
$
\begin{align*}
    <m> &= \dfrac{1}{N\beta}\dfrac{1}{Z} \int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} e^{\kappa(\beta, \phi)} N\beta \tanh(\beta h \pm \phi) \\
    &= \dfrac{1}{Z} \int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} e^{\kappa(\beta, \phi)} \tanh(\beta h \pm \phi)
\end{align*}
$

Comparing this to the expression for the expectation value for some operator $O$, we see that:

$\begin{equation*}
m[\phi] = \tanh(\beta h \pm \phi)
\end{equation*}$

In a similar fashion, we can evaluate the average energy per site. We have the following expression for $<\epsilon>$ as such:

$
\begin{equation*}
<\epsilon> = -\dfrac{1}{N} \dfrac{\partial}{\partial \beta}\left(\log Z \right)  = -\dfrac{1}{NZ} \dfrac{\partial Z}{\partial \beta} 
\end{equation*}
$

Now using the expression for the partition function (with $J > 0$ ) given in the exercise sheet, we have:
$\begin{align*}
    \dfrac{\partial Z}{\partial \beta} &= \dfrac{\partial}{\partial \beta} \left( \int\limits_{-\infty}^{+\infty} \dfrac{d \phi}{\sqrt{2\pi\beta\hat{J}}} \exp{\left[ -\frac{\phi^2}{2\beta\hat{J}} + N\log\left( 2 \cosh(\beta h \pm \phi)\right)\right]}\right) \\
    &=\int\limits_{-\infty}^{+\infty} d \phi \dfrac{\partial}{\partial \beta} \left(\dfrac{\exp{\left[ -\frac{\phi^2}{2\beta\hat{J}} + N\log\left( 2 \cosh(\beta h \pm \phi)\right)\right]}}{\sqrt{2\pi\beta\hat{J}}} \right) \\
    &=\int\limits_{-\infty}^{+\infty} d\phi \left(\dfrac{e^{\kappa(\beta, h)}}{\sqrt{2\pi\beta \hat{J}}} \dfrac{\partial \kappa}{\partial \beta} + e^{\kappa(\beta, h)}\left(-\dfrac{1}{2\beta \sqrt{2\pi\beta\hat{J}}} \right)\right) \\
    &= \int\limits_{-\infty}^{+\infty}  \dfrac{d\phi e^{\kappa(\beta, h)}}{\sqrt{2\pi\beta \hat{J}}}\left(\dfrac{\partial \kappa}{\partial \beta} -\dfrac{1}{2\beta} \right)
\end{align*}$

Now we observe that:
$
\begin{align*}
    \dfrac{\partial \kappa}{\partial \beta} &= \dfrac{\partial}{\partial \beta}\left(-\frac{\phi^2}{2\beta\hat{J}} + N\log\left( 2 \cosh(\beta h \pm \phi)\right)\right) \\
    &= \frac{\phi^2}{2\beta^2\hat{J}} + N\dfrac{2 \sinh(\beta h \pm \phi)}{2 \cosh(\beta h \pm \phi)} h \\
    &= \frac{\phi^2}{2\beta^2\hat{J}} + N h \tanh(\beta h \pm \phi)
\end{align*}
$

Thus combining our expressions, we have the following:
$
\begin{align*}
    <\epsilon> &= -\dfrac{1}{NZ} \int\limits_{-\infty}^{+\infty}  \dfrac{d\phi e^{\kappa(\beta, h)}}{\sqrt{2\pi\beta \hat{J}}}\left(\frac{\phi^2}{2\beta^2\hat{J}} + N h \tanh(\beta h \pm \phi) -\dfrac{1}{2\beta} \right) \\
    &= \dfrac{1}{Z} \int\limits_{-\infty}^{+\infty}  \dfrac{d\phi e^{\kappa(\beta, h)}}{\sqrt{2\pi\beta \hat{J}}}\left(\frac{\phi^2}{2\beta^2N\hat{J}} + h \tanh(\beta h \pm \phi) -\dfrac{1}{2\beta N} \right)
\end{align*}
$

Thus we see that the average energy per site is given as:
$
\begin{equation*}
\epsilon[\phi] = \frac{\phi^2}{2\beta^2N\hat{J}} + h \tanh(\beta h \pm \phi) -\dfrac{1}{2\beta N}
\end{equation*}
$






