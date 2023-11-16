import numpy as np
import sympy
import scipy
import random 
import time
import matplotlib.pyplot as plt
import math
import scipy

class Data_maker:
    def __init__(self,seed:int) -> None:
        random.seed(seed)        
        np.random.seed(seed)
    
        
    
    def random_point2d(self,dim:int,func,bias_interval,x_interval:list,mode:str='random')->tuple:
        '''
        return x_list,y_list,len is dim;\n
        mode='random'or'normal' 
        '''
        seed_list=[]
        out_list=[]
        if mode=='random':
            for i in range(dim):
                seed=random.uniform(x_interval[0],x_interval[1])
                seed_list.append(seed)
                bias=random.uniform(-1/2*bias_interval,1/2*bias_interval)
                out_list.append(func(seed)+bias)
        elif mode=='normal':
            step=(x_interval[1]-x_interval[0])/dim
            seed=x_interval[0]
            for i in range(dim):
                seed_list.append(seed)
                bias=random.uniform(-1/2*bias_interval,1/2*bias_interval)
                out_list.append(func(seed)+bias)
                seed+=step
        return seed_list,out_list
    
    class random_point:
        """generate points uniform destributed in independent dims,dependent dim generate by func, return points in np.ndarray

            Args:
                dim (int): the row of out_mat, the dimension of eigens\n
                nums (int): the col of out_mat, the numbers of samples\n
                scope (tuple): the scope of uniform generate\n
                correlation (list | None):[[dim1,dim2...input_dim],[dim0,dim3...output_dim],func]\n
                    NOTICE:func must input ndarray and output ndarray\n
                    NOTICE: list contains list contains list at least\n
                    NOTICE: if input_dim=output_dim, then will use func to generate axis
                    e.g.:\n
                    [[0,1],[2,3],myaddfunc]\n
                
                noise (list | None): [[[dim0,dim1...],"type",(arg1,arg2...)],[[dim3,dim5...],"type",(arg1,arg2...)]]\n
                    NOTICE: custom_func is INVERSE of distribution function of X,input and output must be np.ndarray\n
                    NOTICE: list contains list contains list at least\n
                    NOTICE: each outlier add to origin, not replace,so that noise can add too!!!\n
                    e.g.\n
                    [[[0,1],"normal",(0=u,1=thegma^2)] , [[0],"uniform",(-1=a,1=b)] ]\n
                    [[[0,1,2],"pulse",(0.1=percent of outliers,(10,20)=abs_scope)] ,[[0,1],"custom",func]]\n
            """
        def __init__(self,
                    dim:int,
                    nums:int,
                    scope:tuple,
                    correlation:list|None=None,
                    noise:list|None=None,
                    )->np.ndarray:
            self.scope=scope
            self.nums=nums
            if correlation==None:
                self.out=np.random.uniform(*scope,(dim,nums))
            else:
                
                out=np.random.uniform(*scope,(dim,nums))
                for each in correlation:
                        
                    input_dim_array=np.array(each[0])
                    output_dim_array=np.array(each[1])
                    if input_dim_array.all()==output_dim_array.all() and len(input_dim_array)==1:
                        out[output_dim_array]=self.make_axis()
                    else:
                        out[output_dim_array]=each[2](out[input_dim_array])
                if noise is not None:
                    for each in noise:
                        dim_array=np.array(each[0])
                        if each[1]=="normal":
                            u=each[2][0]
                            thegma=np.sqrt(each[2][1])
                            out[dim_array]=out[dim_array]+(u+thegma*np.random.randn(len(dim_array),nums))
                        elif each[1]=="uniform":
                            low=each[2][0]
                            high=each[2][1]
                            out[dim_array]=out[dim_array]+np.random.uniform(low,high,(len(dim_array),nums))
                        elif each[1]=="pulse":
                            outlier_nums=round(each[2][0]*nums)
                            outlier_scope=each[2][1]
                            outlier=np.random.uniform(outlier_scope[0],outlier_scope[1],outlier_nums)
                            
                            negative_nums=round(np.random.rand()*outlier_nums)
                            negative_indices=np.random.choice(len(outlier),negative_nums,replace=False)
                            outlier[negative_indices]=-outlier[negative_indices]
                            
                            
                            
                            flatten_matrix=out[dim_array].flatten()
                            outlier_indices=np.random.choice(len(flatten_matrix),outlier_nums,replace=False)
                            flatten_matrix[outlier_indices]=flatten_matrix[outlier_indices]+outlier
                            out[dim_array]=flatten_matrix.reshape(out[dim_array].shape)
                            
                        elif each[1]=="custom":
                            inverse_distribution=each[2]
                            out[dim_array]=out[dim_array]+inverse_distribution(np.random.rand(len(dim_array),nums))
                            
                            
                self.out=out
                
                
        
        def key_close(self,event):
                if event.key=="escape":
                    plt.close(event.canvas.figure)
                
                
        def scatter2d(self,position:int=111,xlim:tuple|None=None,ylim:tuple|None=None,figsize=(10,10)):
            """x is dim0, y is dim1,return figure\n
            You need to show manually

            Args:
                xlim (tuple | None, optional): _description_. Defaults to None.
                ylim (tuple | None, optional): _description_. Defaults to None.
                figsize (tuple, optional): _description_. Defaults to (10,10).
            """
            f=plt.figure(figsize=figsize)
            f.canvas.mpl_connect("key_press_event",self.key_close)
            ax=f.add_subplot(position)
            ax.scatter(self.out[0],self.out[1])
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xlabel("dim0")
            ax.set_ylabel("dim1")
            plt.show()
            return f
        
        def scatter3d(self,position:int=111,xlim:tuple|None=None,ylim:tuple|None=None,zlim:tuple|None=None,figsize=(10,10)):
            """x is dim0,y is dim1, z is dim2,return figure\n
            You need to show manually\n
            Args:
                xlim (tuple | None, optional): _description_. Defaults to None.
                ylim (tuple | None, optional): _description_. Defaults to None.
                figsize (tuple, optional): _description_. Defaults to (10,10).
            """
            f=plt.figure(figsize=figsize)
            f.canvas.mpl_connect("key_press_event",self.key_close)
            ax=f.add_subplot(projection='3d')

            ax.scatter(self.out[0],self.out[1],self.out[2],c='b')
            ax.set_xlabel("dim0")
            ax.set_ylabel("dim1")
            ax.set_zlabel("dim2")
            plt.show()
            return f
            
            
        def make_axis(self):
            return np.linspace(self.scope[0],self.scope[1],self.nums)


    def make_taylor_basis(self,column:np.ndarray,order:int=3):
        column=np.copy(column)
        out=np.ones_like(column)
        for power_index in range(1,order+1):
            out=np.c_[out,np.power(column,power_index)]   
        return out

    def make_fourier_basis(self,column:np.ndarray,order:int=100):
        column=np.copy(column)
        out=np.ones_like(column)    #simulate cos0x
        for multiple_index in range(1,order):
            out=np.c_[out,np.cos(multiple_index*column)]
            out=np.c_[out,np.sin(multiple_index*column)]
        return out
    
         
def least_square():
    def func0(x:np.ndarray)->np.ndarray:
        return np.power(x,2)
    
    def func1(x:np.ndarray)->np.ndarray:
        return np.exp(x)
    def func2(x:np.ndarray):
        return np.log(np.abs(x))
    def func3(x):
        return np.sin(x)
    def func4(x):
        return np.power(x,3)
    data_maker=Data_maker(seed=int(time.perf_counter()))
    point_data=data_maker.random_point(2,1000,(-4,4),
                            [[[0],[0]],[[0],[1],func1]],
                            [[[1],"normal",(0,5)]]
                            )
    
    
    #Ax=y
    A=data_maker.make_taylor_basis(point_data.out[0],order=3)
    y=np.copy(point_data.out[1]).reshape(-1,1)

    t1=time.perf_counter()
    #A@pinv(A.T@A)@A.Ty=x_hex
    
    x_hex=np.linalg.pinv(A.T@A)@A.T@y
    y_hex=A@x_hex
    t2=time.perf_counter()
    
    print("timeis",t2-t1)
    
    #predict using best coefficient_array:x_hex;oder must be same
    axis_predict=np.linspace(4,6,1000)
    B=data_maker.make_taylor_basis(axis_predict,order=3)
    y_predict=B@x_hex
    
    
    
    figure=plt.figure(figsize=(10,10))
    figure.canvas.mpl_connect("key_press_event",point_data.key_close)
    ax=figure.add_subplot(111)
    
    ax.plot(point_data.out[0],y_hex,color='red',linewidth=1)
    ax.scatter(point_data.out[0],point_data.out[1],color='blue')
    ax.plot(axis_predict,y_predict,color='green',linewidth=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()       
           


          
if __name__=="__main__":
    
    
    least_square()
    
    '''def normal_pdf1d(x:np.ndarray):
        return (1/math.sqrt(2*math.pi))*np.exp(-0.5*np.power(x,2))
    def normal_pdf2d(x:np.ndarray):
        
        return (1/math.sqrt(2*math.pi))*np.exp(-0.5*(np.power(x[0],2)+np.power(x[1],2)))
    def power2d(x):
        return np.power(x[0],2)+np.power(x[1],2)
    
    data_maker=Data_maker(10)
    point_data=data_maker.random_point(3,5000,(-3,3),[[[0,1],[2],power2d]])
    point_data.scatter3d() '''
    
    
    