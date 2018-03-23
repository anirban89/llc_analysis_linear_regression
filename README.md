# LINEAR REGRESSION llc velocities

# llc velocity analysis

## The Aim
The aim is to train a deep neural net to infer velocities from the Sea surface heights ($\eta$) and Wind stress($\tau_x$ and $\tau_y$). The training dataset will be the high resolution model, We will then coarsegrain the model fields to get the lower resolution fields and use that as the testing data. The ultimate aim will be to get the velocity fields for Satellite altimetry. 

The hypothesis to be tested is the following:
Can we train a Conv Neural Net to give velocity estimates from altimetry data to get a better signture of small scale (balanced/unbalanced motions) than geostrophy?

## The present work

In this notebook we calculate the surface velocity from the llc4320 model output in the Agulhas sector and write down the formalism for calculating the surface geostrophic velocities from the SSH ($\eta$) and the surface Ekman velocities from wind stress and the formalism for calculating the error.


The total momentum equation can be written as:

![Eqn 1](http://latex.codecogs.com/gif.latex?%24%24%20%5Cfrac%7B%5Cpartial%20%5Cmathbf%7Bu%7D%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Cmathbf%7Bu%7D%20%5Ccdot%20%5Cnabla%20%5Cmathbf%7Bu%7D%20&plus;%20f%20%5Ctimes%20%5Cmathbf%7Bu%7D%20%3D%20-g%20%5Cnabla%20%5Ceta%20&plus;%20%5Cmathbf%7BF%7D%24%24)

where $\mathbf{F}$ is the frictional term

Our traditional method involves splitting the surface flow into a geostrophic and an ageostrophic part as follows:

![Eqn 2](http://latex.codecogs.com/gif.latex?%24%24%5Cmathbf%7Bu%7D%20%3D%20%5Cmathbf%7Bu_g%7D%20&plus;%20%5Cmathbf%7Bu_a%7D%24%24)

where the force balances are 

![Eqn 2](http://latex.codecogs.com/gif.latex?%24%24%20f%20%5Ctimes%20%5Cmathbf%7Bu_g%7D%20%3D%20-g%20%5Cnabla%20%5Ceta%24%24)

and
 ![](http://latex.codecogs.com/gif.latex?%24%24%20f%20%5Ctimes%20%5Cmathbf%7Bu_a%7D%20%3D%20F%24%24)

# Geostrophic velocities

Geostrophic velocities are given by 

$$fv_{g} = g \frac{\partial \eta}{\partial x} $$

$$fu_{g} = - g \frac{\partial \eta}{\partial y} $$



# Ekman velocity 


Under steady state conditions is can be shown that in the boundary layer of the upper ocean (order hundred meters) horizontal gradients are small compared to vertical gradients. Under these conditions, there is a balance between Coriolis and Friction.
Friction in the upper layer is provided by the wind stress.

$$ f v + \frac{\partial \tau_x}{\partial z} = 0$$

$$ f u - \frac{\partial \tau_y}{\partial z} = 0$$ (1)

and 

$$\tau_x = \rho A_z \frac{\partial u}{\partial z}$$

$$\tau_y = \rho A_z \frac{\partial v}{\partial z}$$ (2)


we write $[u,v]$ as a complex velocity as:
$$ \mathbf{u} = u + iv$$

We also write $[\tau_x, \tau_y]$ as a complex wind stress as:
$$ \mathbf{\tau} = \tau_x + i \tau_y$$

This gives us 
$$ \mathbf{u} = u + iv = (u,v)$$
$$ \mathbf{\underline{u}} = -v + iu = (-v,u)$$

$$ \mathbf{\tau} = \tau_x + i\tau_y = (\tau_x,\tau_y)$$
$$ \mathbf{\underline{\tau}} = -\tau_y + i\tau_x = (-\tau_y,\tau_x)$$

therefore the above equations can be written as 

$$  f \mathbf{u} = \frac{\partial }{\partial z} \mathbf{\underline{\tau}}$$

$$ \mathbf{\tau} = \rho A_z \frac{\partial \mathbf{u}}{\partial z} $$

So now using equations (2)

we re write the euqtions as

$$ f \mathbf{u}_{zz} = \frac{ i f }{A_z} \mathbf{u} $$

In the Northern Hemisphere $f > 0$. So the general solution is

$$ \mathbf{u} = \alpha_{+} e^{[(if/A_z)^{1/2}] z} + \alpha_{-} e^{[-(if/A_z)^{1/2}] z} $$

subject to the boundary conditions

$$ A_z {(u + i v)}_z = \tau_x^0 + i \tau_y^0 $$ at $z = 0$

and 

$$ (u + i v) = u_g + i v_g $$  at $z = -H$

For the Northern Hemisphere ($f>0$), the solution is written as

$$ \mathbf{u} = \alpha_{+} e^{[1+i] z/d} + \alpha_{-} e^{-[1+i] z/d} $$

Where the Ekman depth $d$ is

$$ d = \sqrt{\frac{2 A_z}{|f|}}$$

and $\alpha_{+}$ and $\alpha_{-}$ are complex coefficients. the $\alpha_{+}$ part denotes the solution decaying away from the top and the $\alpha_{-}$ denotes the solution decaying away from the bottom. For the surface Ekman velocities we are concerned with the $\alpha_{+}$ part.

For the Southern hemisphere the solution now becomes

$$ \mathbf{u} = \alpha_{+} e^{[-1+i] z/d} + \alpha_{-} e^{-[-1+i] z/d} $$

Here now the $\alpha_{-}$ part is the solution for the surface Ekman flow.

### Northern Hemisphere 
To solve for $\alpha_{+}$, plug in $\mathbf{u}$ in the equation 

$$ \mathbf{\tau} = \rho A_z \frac{\partial \mathbf{u}}{\partial z} $$

This gives us at $z = 0$

$$ \alpha_{+} = \frac{(\tau_x +\tau_y) d}{\rho A_z (1+i)} = \frac{(\tau_x +\tau_y) (1-i) d}{2 \rho A_z }$$

$$ \implies \alpha_{+} = \frac{(\mathbf{\tau} - \mathbf{\underline{\tau}}) d}{2 \rho A_z}$$

So, we have

$$ u_{e} + i v_{e} = \frac{d}{2 \rho A_z} \left[ (\tau_x + i\tau_y) - (-\tau_y + i\tau_x)\right]$$

$$ \implies u_{e} = \frac{1}{\rho \sqrt{2 A_z |f|}} (\tau_x + \tau_y)$$
$$ \implies v_{e} = \frac{1}{\rho \sqrt{2 A_z |f|}} (-\tau_x + \tau_y)$$



### Southern Hemisphere 

Doing the similar procedure in the Southern Hemisphere we get

$$ \implies \alpha_{-} = \frac{(\mathbf{\tau} + \mathbf{\underline{\tau}}) d}{2 \rho A_z}$$


So, we have

$$ u_{e} + i v_{e} = \frac{d}{2 \rho A_z} \left[ (\tau_x + i\tau_y) + (-\tau_y + i\tau_x)\right]$$

$$ \implies u_{e} = \frac{1}{\rho \sqrt{2 A_z |f|}} (\tau_x - \tau_y)$$
$$ \implies v_{e} = \frac{1}{\rho \sqrt{2 A_z |f|}} (\tau_x + \tau_y)$$
 

# Linear Regression 

For the first exercise, we aim to fit a multiple linear regression. For our black box therefore the input variables are $\left[x_{i1}, ... , x_{ip}\right]_{i=1}^{n}$, n being the number of samples, and p being the number of features. We can represent the linear regression problem as $u_i = \beta_0 1 + \beta_1 x_{i1} + ... + \beta_p x_{ip} + \epsilon_i$.

$$
X=
  \begin{bmatrix}
    1 & x_{11} & x_{12} ...  & x_{1p} \\
    1 & x_{21} & x_{22} ...  & x_{2p} \\
    .. & ..  & .... &...\\
    1 & x_{n1} & x_{n2} ...  & x_{np}
  \end{bmatrix}
$$

or 

$$
X = \begin{bmatrix}
    x_1^T \\
    x_2^T \\
    .. \\
    x_n^T
  \end{bmatrix}
$$

and 

$$
\beta = \begin{bmatrix}
    \beta_0 \\
    \beta_1 \\
    .. \\
    \beta_p
  \end{bmatrix}
$$

$$
U= \begin{bmatrix}
    u_0 \\
    u_1 \\
    .. \\
    u_n
  \end{bmatrix}
$$

$$
\epsilon = \begin{bmatrix}
    \epsilon_0 \\
    \epsilon_1 \\
    .. \\
    \epsilon_n
  \end{bmatrix}
$$

For our example our 9 features (variables) are $[f, \tau_x,\tau_y, \eta_{x+}, \eta_{x-}, \eta_{y+}, \eta_{y-}, \frac{1}{dx}, \frac{1}{dy}]$ .

where $U$ and $\beta$ are complex valued. For vectorization we consider them to have 2 columns each

$$
U = \begin{bmatrix}
    u_1 & v_1\\
    u_2 & v_2\\
    .. \\
    u_n & v_n
  \end{bmatrix}
$$

and 

$$
\beta = \begin{bmatrix}
        \beta_{0r} & \beta_{0i}\\
        \beta_{1r} & \beta_{1i}\\
        .. \\
        \beta_{pr} & \beta_{pi}
  \end{bmatrix}
$$

Our linear regression problem is therefore 

![](http://latex.codecogs.com/gif.latex?%24%24%20%5Cunderbrace%7B%5Cbegin%7Bbmatrix%7D%20u_1%20%26%20v_1%5C%5C%20u_2%20%26%20v_2%5C%5C%20..%20%5C%5C%20u_n%20%26%20v_n%20%5Cend%7Bbmatrix%7D%7D_%7Bu%20%3D%20%5Bn%20%28samples%29%20%5Ctimes%202%5D%7D%20%3D%20%5Cunderbrace%7B%5Cbegin%7Bbmatrix%7D%201%20%26%20x_%7B11%7D%20%26%20x_%7B12%7D%20...%20%26%20x_%7B19%7D%20%5C%5C%201%20%26%20x_%7B21%7D%20%26%20x_%7B22%7D%20...%20%26%20x_%7B29%7D%20%5C%5C%20..%20%26%20..%20%26%20....%20%26...%5C%5C%201%20%26%20x_%7Bn1%7D%20%26%20x_%7Bn2%7D%20...%20%26%20x_%7Bn9%7D%20%5Cend%7Bbmatrix%7D%7D_%7BX%20%3D%20%5Cleft%5Bn%20%28samples%29%20%5Ctimes%20%5B9%28features%29&plus;1%5D%5Cright%5D%7D%20%5Ccdot%20%5Cunderbrace%7B%5Cbegin%7Bbmatrix%7D%20%5Cbeta_%7B0r%7D%20%26%20%5Cbeta_%7B0i%7D%5C%5C%20%5Cbeta_%7B1r%7D%20%26%20%5Cbeta_%7B1i%7D%5C%5C%20..%20%5C%5C%20%5Cbeta_%7B9r%7D%20%26%20%5Cbeta_%7B9i%7D%20%5Cend%7Bbmatrix%7D%7D_%7B%5Cbeta%20%3D%20%5Cleft%5B%5B9%28coefficients%29&plus;1%28intercept%29%5D%20%5Ctimes%202%5Cright%5D%7D%20%24%24)


here $[\beta_1, ..., \beta_9]$ are the coefficients and $\beta_0$ is the intercept.

or 

![](http://latex.codecogs.com/gif.latex?%5Cinline%20%24%24%20%5Cmathbf%7Bu%7D%20%3D%20%5Cbeta_0%20&plus;%20%5Cbeta_1%20%5Ccdot%20%5Cunderbrace%7B%5Bf_%7Bx%2Cy%7D%5D%7D_%7Bx_1%7D%20&plus;%20%5Cbeta_2%20%5Ccdot%20%5Cunderbrace%7B%5B%5Ctau%5Ex_%7Bx%2Cy%7D%5D%7D_%7Bx_2%7D%20&plus;%20%5Cbeta_3%20%5Ccdot%20%5Cunderbrace%7B%5B%5Ctau%5Ey_%7Bx%2Cy%7D%5D%7D_%7Bx_3%7D%20&plus;%20%5Cbeta_4%20%5Ccdot%20%5Cunderbrace%7B%5B%5Ceta_%7Bx&plus;%2Cy%7D%5D%7D_%7Bx_4%7D%20&plus;%20%5Cbeta_5%20%5Ccdot%20%5Cunderbrace%7B%5B%5Ceta_%7Bx-%2Cy%7D%5D%7D_%7Bx_5%7D%20&plus;%20%5Cbeta_6%20%5Ccdot%20%5Cunderbrace%7B%5B%5Ceta_%7Bx%2Cy&plus;%7D%5D%7D_%7Bx_6%7D%20&plus;%20%5Cbeta_7%20%5Ccdot%20%5Cunderbrace%7B%5B%5Ceta_%7Bx%2Cy-%7D%5D%7D_%7Bx_7%7D%20&plus;%20%5Cbeta_8%20%5Ccdot%20%5Cunderbrace%7B%5B%5Cfrac%7B1%7D%7Bdx%7D_%7Bx%2Cy%7D%5D%7D_%7Bx_8%7D%20&plus;%20%5Cbeta_9%20%5Ccdot%20%5Cunderbrace%7B%5B%5Cfrac%7B1%7D%7Bdy%7D_%7Bx%2Cy%7D%5D%7D_%7Bx_9%7D%20%24%24)


# Our insight 

Based on our null hypothesis that 
$$ \mathbf{u} = \mathbf{u_g} + \mathbf{u_{ek}}$$

$$\mathbf{u} = 
\frac{-g}{2} \left([\eta_{x,y+}] - [\eta_{x,y-}]\right) [\frac{1}{dy_{x,y}}] [f_{x,y}]^{-1} + 
\frac{1}{\rho \sqrt{2 A_z}} [f_{x,y}]^{-1/2}\left([\tau^x_{x,y}] - [\tau^y_{x,y}] \right) + 
\hat{i} \frac{g}{2} \left([\eta_{x+,y}] - [\eta_{x-,y}]\right) [\frac{1}{dx_{x,y}}] [f_{x,y}]^{-1} + 
\hat{i} \frac{1}{\rho \sqrt{2 A_z}} [f_{x,y}]^{-1/2}\left([\tau^x_{x,y}] + [\tau^y_{x,y}] \right)
$$

$$\mathbf{u} = 
\frac{-g}{2} \left( [x_6] - [x_7] \right) [x_9] [x_1]^{-1} + 
\frac{1}{\rho \sqrt{2 A_z}} [x_1]^{-1/2}\left([x_2] - [x_3] \right) + 
\hat{i} \frac{g}{2} \left([x_4] - [x_5]\right) [x_8] [x_1]^{-1} + 
\hat{i} \frac{1}{\rho \sqrt{2 A_z}} [x_1]^{-1/2}\left([x_2] + [x_3] \right)
$$

$$
\mathbf{u} = 
[-c_1,0] \underbrace{(x_1^{-1}x_6 x_9)}_{y_1} + 
[c_1, 0] \underbrace{(x_1^{-1} x_7 x_9)}_{y_2} + 
[0,c_1] \underbrace{(x_1^{-1}x_4 x_8)}_{y_3} + 
[0,-c_1] \underbrace{(x_1^{-1} x_5 x_8)}_{y_4} +
[c2, c2] \underbrace{(x_1^{-1/2} x_2)}_{y_5} + 
[-c_2, c_2] \underbrace{(x_1^{-1/2} x_3)}_{y_6}
$$

where $c_1 =\frac{g}{2}$ and $c_2 = \frac{1}{\rho \sqrt{2 A_z}}$ 
This represents our null hypothesis where the coefficients are 
$$
\theta = \begin{bmatrix}
        -c_1 & 0\\
        c_1 & 0\\
        0 & c_1\\
        0 & -c_1\\
        c_2 & c_2\\
        -c_2 & c_2
  \end{bmatrix}
$$

The new linear regression problem now becomes

![](http://latex.codecogs.com/gif.latex?%5Cinline%20%24%24%5Cmathbf%7Bu%7D%20%3D%20%5Ctheta_0%20&plus;%20%5Ctheta_1%20%5Cunderbrace%7B%28x_1%5E%7B-1%7Dx_6%20x_9%29%7D_%7By_1%7D%20&plus;%20%5Ctheta_2%20%5Cunderbrace%7B%28x_1%5E%7B-1%7D%20x_7%20x_9%29%7D_%7By_2%7D%20&plus;%20%5Ctheta_3%20%5Cunderbrace%7B%28x_1%5E%7B-1%7Dx_4%20x_8%29%7D_%7By_3%7D%20&plus;%20%5Ctheta_4%20%5Cunderbrace%7B%28x_1%5E%7B-1%7D%20x_5%20x_8%29%7D_%7By_4%7D%20&plus;%20%5Ctheta_5%20%5Cunderbrace%7B%28x_1%5E%7B-1/2%7D%20x_2%29%7D_%7By_5%7D%20&plus;%20%5Ctheta_6%20%5Cunderbrace%7B%28x_1%5E%7B-1/2%7D%20x_3%29%7D_%7By_6%7D%20%24%24)


Or we can write in vector form as

![](http://latex.codecogs.com/gif.latex?%5Cinline%20%24%24%20%5Cunderbrace%7B%5Cbegin%7Bbmatrix%7D%20u_1%20%26%20v_1%5C%5C%20u_2%20%26%20v_2%5C%5C%20..%20%5C%5C%20u_n%20%26%20v_n%20%5Cend%7Bbmatrix%7D%7D_%7Bu%20%3D%20%5Bn%20%28samples%29%20%5Ctimes%202%5D%7D%20%3D%20%5Cunderbrace%7B%5Cbegin%7Bbmatrix%7D%201%20%26%20y_%7B11%7D%20%26%20y_%7B12%7D%20...%20%26%20y_%7B16%7D%20%5C%5C%201%20%26%20y_%7B21%7D%20%26%20y_%7B22%7D%20...%20%26%20y_%7B26%7D%20%5C%5C%20..%20%26%20..%20%26%20....%20%26...%5C%5C%201%20%26%20y_%7Bn1%7D%20%26%20y_%7Bn2%7D%20...%20%26%20y_%7Bn6%7D%20%5Cend%7Bbmatrix%7D%7D_%7BY%20%3D%20%5Cleft%5Bn%20%28samples%29%20%5Ctimes%20%5B6%28features%29&plus;1%5D%5Cright%5D%7D%20%5Ccdot%20%5Cunderbrace%7B%5Cbegin%7Bbmatrix%7D%20%5Ctheta_%7B0r%7D%20%26%20%5Ctheta_%7B0i%7D%5C%5C%20%5Ctheta_%7B1r%7D%20%26%20%5Ctheta_%7B1i%7D%5C%5C%20..%20%5C%5C%20%5Ctheta_%7B6r%7D%20%26%20%5Ctheta_%7B6i%7D%20%5Cend%7Bbmatrix%7D%7D_%7B%5Cbeta%20%3D%20%5Cleft%5B%5B6%28coefficients%29&plus;1%28intercept%29%5D%20%5Ctimes%202%5Cright%5D%7D%20%24%24)




