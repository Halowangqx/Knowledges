# Mathematical operation

## outer

*"outer" refers to operations that involve pairwise combinations of elements from two input sets or arrays, producing a new set or array.*

In the case of "**outer addition,**" the term describes an operation where you take two input arrays and compute the pairwise addition of their elements, resulting in a new array with a shape equal to the concatenation of the input shapes. The term "outer" is used here because the operation involves combining elements from the "outer" dimensions of the input arrays.

Similarly, you may have heard of the "**outer product**" in the context of vectors and matrices. The outer product of two vectors creates a matrix, where each element at position (i, j) is the product of the i-th element of the first vector and the j-th element of the second vector. This operation is also called the "tensor product" in some contexts.



# Special Orthogonal group (SO)

orthogonal matrices with determinant equal to 1



## SO(3)

SO(3) stands for the "Special Orthogonal group of order 3" and represents the set of all 3x3 **orthogonal matrices with determinant equal to 1**. In simpler terms, SO(3) transformations describe rotations in three-dimensional space. **These transformations preserve the distance between points and the angles between vectors**, which makes them particularly important in fields such as robotics, computer vision, and computer graphics.

*If the determinant were -1, it would mean that the transformation includes a reflection, which would change the orientation (handedness) of the coordinate system.*





## Rotation

Geometry provides us with four types of transformations, namely, rotation, reflection, translation, and resizing.



### Rotation Matrices

- rotate a vector in the counterclockwise direction by an angle θ.

- [orthogonal matrices](https://www.cuemath.com/algebra/orthogonal-matrix/) with a determinant equal to 1



> basic/elementary rotation

```python
##### row
# counterclockwise rotation of \theta about the x axis
R_x(θ) = | 1    0         0   |
         | 0  cos(θ)  -sin(θ) |
         | 0  sin(θ)   cos(θ) |
        
##### pitch
# counterclockwise rotation of \theta about the y axis
R_y(θ) = |  cos(θ)  0  sin(θ) |
         |    0     1    0    |
         | -sin(θ)  0  cos(θ) |
        
##### yaw
# counterclockwise rotation of \theta about the z axis
R_z(θ) = | cos(θ)  -sin(θ)  0 |
         | sin(θ)   cos(θ)  0 |
         |   0        0     1 |
```

**The columns of R represent the transformed basis vectors of the rotated coordinate system**

> 旋转的本质是，将原来的基 $(e_1, e_2, ... , e_n)$ 转化成了一组新的正交基
>
> column vectors ---> new basis ---> new axises

**the rows represent the original coordinate system's basis vectors projected onto the rotated coordinate system.**



### Euler angles

Euler angles are a set of three angles (α, β, γ) that describe a rotation in 3D space using three consecutive rotations about different axes. There are multiple conventions for Euler angles, such as ZYX (yaw-pitch-roll), XYZ (roll-pitch-yaw), and others. Euler angles are compact and intuitive but suffer from gimbal lock, a phenomenon that can cause loss of a degree of freedom in certain configurations.



**Problems**:

- gimbal lock



> In summary, the correspondence between rotation matrices and Euler angles is one-to-one for a specific order, except in cases of gimbal lock, where multiple Euler angle combinations can represent the same rotation matrix.





### Axis-angle representation

$$
\mathbf{\theta} = \theta e
$$

Axis-angle representation uses a unit vector (k_x, k_y, k_z) to define the rotation axis and an angle θ to define the rotation magnitude around that axis. The axis-angle representation avoids gimbal lock and has a more compact representation than rotation matrices but requires conversion to other representations for certain operations, such as interpolation or composition.



### Quaternions

Quaternions are a four-dimensional extension of complex numbers and can represent rotations in 3D space using a scalar part (w) and a vector part (x, y, z). Quaternions avoid gimbal lock, have a compact representation, and can be easily interpolated (e.g., using SLERP for smooth rotation interpolation). However, they are less intuitive than Euler angles and require conversions for certain operations.

Each representation has its own specific use cases, advantages, and drawbacks. In general, SO(3) transformations are crucial for representing and manipulating rotations in 3D space across various applications.

#### Stereographic Projection

**help us to figure out what's a quaternion be like in a 4-d space**

通过这种投影，可以将一个n维的东西投影到n-1维上

- 复平面(2-d)上的单位圆 ---> vertical line (1-d)
- a uni-sphere (3-d) ---> a plane (2-d)
- a uni-hypersphere (4-d) ---> a whole 3-d space





在投影时，

- real-part --> omitted --> 改变了在投影空间中的位置（scale）
- virtual-part --> be projected on --> 改变了在投影空间中的方向（orientation）



2-d rotation --> 1-degree freedom --> uni-complex

3-d rotation --> 3-degree freedom --> uni-quaternion



#### What is quaternion

- Definition:

$$
q = w +x\boldsymbol{i}+y\boldsymbol{j}+k\boldsymbol{k}
$$



- how to multiply?

  - compute the production

  $$
  \begin{align}
  For\ i,j,k:\\
  ij&= -ji=k\\
  jk&=-kj=i\\
  ki&=-ik=j
  \end{align}
  $$

  - scale or squish

$$
q_1 \cdot q_2 = \underbrace{\frac{q_1}{\Vert{q_1}\Vert}}_{Apply\ special\ 4d\ rotation} \overbrace{\Vert q_1\Vert \cdot q_2}^{Scale\ q_2}
$$

#### quaternion and 3-d rotation

- 

- **double cover**

  each rotation in 3-d corresponds to 2 quaternions



# Implicit function

## Representation of 3D learning

> The performance of machine learning methods is heavily dependent on the choice of data representation (or features) on which they are applied.





- point-based

  light-weight

  closely matches the raw data that many sensors (i.e.
  LiDARs, depth cameras) provide, and hence is a natural fit
  for applying 3D learning. 

  

  do not **desribe topology** and are not suitable for producing **watertight surfaces**

  

   Learning on: *PointNet* style approach 

- Mesh-based

- Voxel-based

  



## DeepSDF

> unlike common surface reconstruction tech-
> niques which discretize this SDF into a regular grid for eval-
> uation and measurement denoising, we instead learn a gen-
> erative model to produce such a continuous field.

**continuous generalizable 3D generative models of SDF**

![Screenshot from 2023-06-25 18-20-18](/home/wangqx/Pictures/Screenshot from 2023-06-25 18-20-18.png)















