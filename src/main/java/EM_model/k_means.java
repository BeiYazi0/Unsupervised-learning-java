package EM_model;


import java.awt.image.BufferedImage;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.index.NDIndex;

import helper.img;


public class k_means {
	NDArray o_array, array, mean; // 原矩阵，k-means可处理的矩阵，集群中心值
	NDManager manager;            // 管理器
	int k;                        // 集群数
	Shape shape;                  // o_array 的形状(高, 宽, 通道数)
	public k_means(BufferedImage bf, int k) {
		// 用于创建 NDArray 的管理器
		this.manager = NDManager.newBaseManager();
		
		int width = bf.getWidth();    // 图片宽度
        int height = bf.getHeight();  //图片高度

        // 原图
		int[][][] rgb3DArray = img.image2array(bf);
		set_o_array(rgb3DArray, height, width);
		
		this.k = k;                         // 集群值
		this.shape = o_array.getShape();    // 原图的形状	
	}
	
	// 将 rgb 数组转换为 NDArray
	public void set_o_array(int[][][] rgb3DArray, int height,int width) {
		int[][] r = new int[height][width];
		int[][] g = new int[height][width];
		int[][] b = new int[height][width];
		
		for(int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                r[i][j] = rgb3DArray[i][j][0];
                g[i][j] = rgb3DArray[i][j][1];
                b[i][j] = rgb3DArray[i][j][2];
            }
        }
		
		// 分别创建 rgb 的 NDArray 并拼接
		NDArray rvalue = manager.create(r).reshape(height, width, 1);
		NDArray gvalue = manager.create(g).reshape(height, width, 1);
		NDArray bvalue = manager.create(b).reshape(height, width, 1);
		NDArray rgvalue = rvalue.concat(gvalue, 2);
		NDArray rgbvalue = rgvalue.concat(bvalue, 2);
		this.o_array = rgbvalue.toType(DataType.FLOAT64, false);
		rvalue.close();gvalue.close();bvalue.close();rgvalue.close();rgbvalue.close();
	}
	
	// 由 NDArray 还原 rgb 矩阵
	public int[][][] get_rgbarray(int height,int width) {
		int[][][] rgb3DArray = new int[height][width][3];
		
		for(int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
            	rgb3DArray[i][j] = array.get(i,j).toIntArray();
            }
        }
		
		return rgb3DArray;
	}
	
    // 选定 axis 进行打乱
	public NDArray shuffle(NDArray array, int k) {
		long[] shape = array.getShape().getShape();      // array 的形状
		long length = shape[k];
		
		int[] index = manager.arange((int)length).toIntArray(); // 待打乱的索引
		for(int i = 0; i < length; i++){
            int iRandNum = (int)(Math.random() * length);
            int temp = index[iRandNum];
            index[iRandNum] = index[i];
            index[i] = temp;
        }
		
		NDArray idx = manager.create(index);
		NDArray newArr = array.swapAxes(0, k).get(idx).swapAxes(0, k);
		idx.close();
		return newArr;
	}
	
	// 初始化 k 个集群的中心值
	public void get_initial_means() {
		NDArray newArr = shuffle(this.array, 0);                // 随机打乱
		this.mean = newArr.get(new NDIndex("0:" + k + ",:"));   // 选择前 k 个点
		newArr.close();
	}
	
	// k-means 单步
	public NDArray k_means_step(int m) {	
		// 样本的分类结果
		NDArray clusters = manager.zeros(new Shape(m));
		int[][] axes = {{0},{1}};
		for(int i = 0; i < m; i++) {
			NDArray curSample = array.get(i);  // 当前的样本
			
			// 计算样本的二范数(样本与各个集群中心值的距离)
			NDArray diff = mean.sub(curSample);
			NDArray norm = diff.norm(axes[1]);
			
			// 聚类，每个样本归类到距离其最近的集群
			NDArray cluster = norm.argMin();
			clusters.set(new NDIndex(i), cluster);
			
			// 回收
			diff.close();
			curSample.close();
			norm.close();
			cluster.close();
		}
		
		// 所有样本的索引
		NDArray index = manager.arange(m);
		// 更新每个集群的新的中心值
		for(int i = 0; i < k; i++) {
			// 属于集群 i 的所有样本的索引
			NDArray mask = clusters.eq(i);
			NDArray cur_index = index.get(mask);
			// 属于集群 i 的所有样本
			NDArray cluster_samples = array.get(cur_index);
			
			// 集群中心值是所有属于该集群的样本的均值
			if(!cluster_samples.isEmpty()) {
				NDArray mean = cluster_samples.mean(axes[0]);
				this.mean.set(new NDIndex(i + ",:"), mean);
				mean.close();
			}
			
			// 回收
			mask.close();
			cur_index.close();
			cluster_samples.close();
		}
		index.close();
		return clusters;
	}
	
	// 训练模型并重构图像
	public BufferedImage segment(int max_iter) {
		// 原图的大小
		int r = (int)shape.getShape()[0];
		int c = (int)shape.getShape()[1];
		int ch = (int)shape.getShape()[2];
		int m = r*c;
		
		// 创建一个 k_means_step 可以处理的矩阵
		Shape arr_shape = new Shape(m, ch);
		this.array = this.o_array.reshape(arr_shape);
		
		// 初始化
		get_initial_means();
		
		NDArray preClusters = manager.zeros(new Shape(m));     // 原集群  
		// 迭代直到收敛或达到最大迭代次数
		for(int i = 0; i < max_iter; i++) {
			System.out.println(i);
			NDArray curClusters = k_means_step(m); // 新的集群
			
			// 收敛时结束迭代
			if(preClusters.contentEquals(curClusters)) {
				break;
			}
			
			preClusters.set(new NDIndex(":"), curClusters);
			curClusters.close();
		}
		
		// 将原始值替换为相应的集群值并重构图像
		this.array.close();
		NDArray updated_array = mean.get(preClusters);
		this.array = updated_array.reshape(shape).toType(DataType.INT32, false);
		int[][][] rgb3DArray = get_rgbarray(r, c);
		BufferedImage update_Bufimg = img.array2Image(rgb3DArray, r, c);
		preClusters.close();
		updated_array.close();
		return update_Bufimg;
	}
	
	// 释放内存
	public void close() {
		this.o_array.close();
		this.array.close();
		this.mean.close();
		this.manager.close();
	}
}
