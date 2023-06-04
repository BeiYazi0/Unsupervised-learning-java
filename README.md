# �޼ලѧϰ

## ǰ��

���˵ĵ�һ��java��Ŀ������һ�¡�

���ֿ�Ϊ[Unsupervised-learning](https://github.com/BeiYazi0/Unsupervised-learning)��ͼǨ�Ƶ�java�İ汾��

��ʵ��k-means���࣬��δʵ�� EM �㷨�ĸ�˹���ģ�͡�

### ��ȡ�ֿ�

```
git clone https://github.com/BeiYazi0/Unsupervised-learning-java
```

## ����

�޼ලѧϰ������һ��û�б�ǩ����û�н��й�����������У���������֮��Ĺ��ԣ�Ȼ����������֮��Ĺ��Զ��µ����ݽ��в��������ǽ�ʵ�������޼ලѧϰ�㷨��k-means �� EM �㷨��

��˹���ģ���� EM �㷨��һ��Ӧ�ã������Դ�һ�������������������ȡ����ͬ��ĸ�˹ģ�ͳ���������ͨ����������ʵ��һ����˹���ģ��:

1. ʵ�� k-means ���ࡣ

2. ͨ���ڼ����ݼ�������������Ϥ�㷨��

3. ����һ�������������ѵ���ĸ�˹���ģ�͡�

4. �Ľ���˹���ģ��ʵ�ֵ�ϸ�ڡ�

5. ʵ��һ���µĶ�������Ҷ˹��Ϣ׼�򣩣��Եõ���³����ģ�͡�

## Prepare: DJL

DJL ���� ��ȫ�� Java ���������ѧϰƽ̨��

�������ݿ�ѧ�������е�Ӧ�������ӣ�ʹ�� Nά���� ���ı�����ݱ��������Ҫ�����ǿ��Խ���ȥ���ݿ�ѧ�����еĶ�άѭ��Ƕ�������Ϊ�򵥼��С����ڽ�һ���ͷ��˼��㲢���������⼸�м򵥵Ĵ��������ٶ�Ҳ��ȴ�ͳ��άѭ����ܶࡣ

������ѧ����İ��Ѿ���Ϊ���ݿ�ѧ��ͼ��ѧ�Լ�����ѧϰ����ı�׼��ͬʱ����Ӱ�������ڲ��ϵ�������������

�� Python �����磬���� NDArray��Nά���飩�ı�׼������ NumPy����������� Java �����У���û����֮ͬ����׼�Ŀ⡣Ϊ�˸� Java �����ߴ���ͬһ��ʹ�û���������ѷ�Ʒ���Դ�� DJL һ������ Java �����ѧϰ�⡣

���������������ѧϰģ�飬����������ĵ� NDArray ϵͳ���Ա����� Nά���� �ı�׼�����߱������Ŀ���չ�ԡ�ȫƽ̨֧���Լ�ǿ��ĺ������֧�� (TensorFlow��PyTorch��Apache MXNet���������� CPU ���� GPU��PC ���ǰ�׿��DJL ����������׾ٵ��������

��Ŀ��ַ��[djl](https://github.com/awslabs/djl/)

�����������ǽ���һ���˽� NDArray������ѧϰ���д�� Numpy ͬ���򵥵� Java ���롣

### ��װ DJL

#### ���� gradle ��Ŀ
```
plugins {
    id 'java'
}
repositories {                           
    jcenter()
}
dependencies {
    implementation "ai.djl:api:0.6.0"
    // PyTorch
    runtimeOnly "ai.djl.pytorch:pytorch-engine:0.6.0"
    runtimeOnly "ai.djl.pytorch:pytorch-native-auto:1.5.0"
}
```

#### ���� Maven ��Ŀ

```
    <dependency>
		<groupId>ai.djl</groupId>
		<artifactId>api</artifactId>
		<version>0.20.0</version>
	</dependency>
	<dependency>
		<groupId>ai.djl.pytorch</groupId>
		<artifactId>pytorch-engine</artifactId>
		<version>0.20.0</version>
		<scope>runtime</scope>
	</dependency>
	<dependency>
		<groupId>ai.djl.pytorch</groupId>
		<artifactId>pytorch-model-zoo</artifactId>
		<version>0.20.0</version>
    </dependency>
```

### ��������

���ȳ��Խ���һ�� try block ���������ǵĴ���

```
try(NDManager manager = NDManager.newBaseManager()) {
}
```

NDManager �� DJL �е�һ�� class ���԰������� NDArray ���ڴ�ʹ�á�ͨ������ NDManager ���ǿ��Ը���ʱ�Ķ��ڴ������������� block ��������������ʱ���ڲ������� NDArray ���ᱻ������������Ʊ�֤�������ڴ��ģʹ�� NDArray �Ĺ����У�����ͨ���������е� NDManager ������Ч�������ڴ档

#### ���� NDArray

`ones `��һ������ȫ��1��Nά�������
```
NDArray nd = manager.ones(new Shape(2, 3));
/*
ND: (2, 3) cpu() float32
[[1., 1., 1.],
 [1., 1., 1.],
]
*/
```
���Գ������������������������Ҫ����һЩ�� 0 �� 1 �������
```
NDArray nd = manager.randomUniform(0, 1, new Shape(1, 1, 4));
/*
ND: (1, 1, 4) cpu() float32
[[[0.932 , 0.7686, 0.2031, 0.7468],
 ],
]
*/
```

#### ��ѧ����

����ʹ�� NDArray ����һϵ�е���ѧ�������������������һ��ת�ò�����Ȼ����������ݼ�һ�����Ĳ��������Բο����µ�ʵ�֣�

```
NDArray nd = manager.arange(1, 10).reshape(3, 3);
nd = nd.transpose();
nd = nd.add(10);
/*
ND: (3, 3) cpu() int32
[[11, 14, 17],
 [12, 15, 18],
 [13, 16, 19],
]
*/
```

DJL ֧�� 60 ���ֲ�ͬ�� NumPy ��ѧ���㣬���������˴󲿷ֵ�Ӧ�ó�����

[Interface NDArray](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/ndarray/NDArray.html)

#### Get �� Set

һ������ NDArray ����Ҫ��������������ɼ򵥵���������/��ȡ���ܡ��ο� NumPy ����ƣ��� Java ��ȥ�������ݱ���е��������˾��򻯴���

������ɸѡһ��Nά��������С��10����

```
NDArray nd = manager.arange(5, 14);
nd = nd.get(nd.gte(10));
/*
ND: (4) cpu() int32
[10, 11, 12, 13]
*/
```

����������һ��һ����΢����һЩ��Ӧ�ó���������������һ��3x3�ľ���Ȼ��������ѵڶ��е����ݶ�����2:

```
NDArray nd = manager.arange(1, 10).reshape(3, 3);
nd.set(new NDIndex(":, 1"), array -> array.mul(2));
/*
ND: (3, 3) cpu() int32
[[ 1,  4,  3],
 [ 4, 10,  6],
 [ 7, 16,  9],
]
*/
```

�� Java ������һ�� NDIndex �� class���������˴󲿷��� NumPy �ж��� NDArray ֧�ֵ� get/set ������ֻ��Ҫ�򵥵ķŽ�ȥһ���ַ������ʽ���������� Java �п���������ת��������Ĳ�����

[Class NDIndex](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/ndarray/index/NDIndex.html)

## Part 0: ��Ŀ�ṹ

����ĿΪ Maven ��Ŀ��ǰ��׼������׸����

jdk�汾��1.8 or above.

��Ŀ��֯�ṹ����ͼ

![](resource/struct.png)

1. `src` ���ڴ����Ŀ����Ͳ��Դ��룬����Ŀ�޲��Դ��롣

2. `JRE System Library` ������JAVA����������Ļ����ļ��ϣ�����JVM��׼ʵ�ּ�Java������⡣

3. `Maven Dependencies` �ǵ�����������

4. `resource` ��ű���Ŀ�������Դ����Ҫ��ͼ��

5. `target` ������ɵ� class �ļ��Լ�����������ɵ� jar ����

6. [pom.xml](pom.xml) �����������������������

��Ŀ������������������ɣ�

### Part 0a: EM_model

EM �㷨ģ�͵�ʵ�֣����� k-means ģ�ͺ͸�˹���ģ�͡�

k_means ��ĳ�Ա�ͷ��������[k_means.java](src/main/java/EM_model/k_means.java)��
```
NDArray o_array, array, mean; // ԭ����k-means�ɴ���ľ��󣬼�Ⱥ����ֵ
NDManager manager;            // ������
int k;                        // ��Ⱥ��
Shape shape;                  // o_array ����״(��, ��, ͨ����)

public k_means(BufferedImage bf, int k); // ���캯��

public void set_o_array(int[][][] rgb3DArray, int height,int width); // �� rgb ����ת��Ϊ NDArray

public int[][][] get_rgbarray(int height,int width); // �� NDArray ��ԭ rgb ����

public NDArray shuffle(NDArray array, int k); // ѡ�� axis ���д���

public void get_initial_means(); // ��ʼ�� k ����Ⱥ������ֵ

public NDArray k_means_step(int m); // k-means ����

public BufferedImage segment(int max_iter); // ѵ��ģ�Ͳ��ع�ͼ��
	
public void close(); // �ͷ��ڴ�
```

gauss_model ��ķ�����k_means ������࣬��δʵ�֣����ܵ���ʽ��
```
initialize_parameters(X, k); // ��ʼ�� k ���㣬���ؾ�ֵ��Э����ͳ�ʼ���ϵ��

prob(X, mu, sigma); // ���ݾ�ֵ��Э���������һ�������ܶȹ���

E_step(X,MU,SIGMA,PI,k); // ���� 1������ÿ����Ⱥ�ĸ����ܶȹ��ƣ����ξ���

M_step(X, r, k); // ���� 2���������ξ������¼����ֵ��Э����ͻ��ϵ��

loglikelihood(X, PI, MU, SIGMA, k); // ����ѵ��ģ�͵Ķ�����Ȼ

train_model(X, k); // ѵ�����ģ�ͣ����� E_step �� M_step ֱ������(��Ȼֵ�ж�)

cluster(r); // �������ξ���õ�ʹ����Ȼֵ���ļ�Ⱥ����(����)

segment(X, MU, k, r); // ÿ�����ݵ㱻�滻Ϊ��Ⱥֵ(�����Ȼ������ֵ)

best_segment(X,k,iters); // ͨ������ѵ��ģ�Ͳ���������Ȼ��ȷ��ͼ�����ѷָ�
```

### Part 0b: helper

���߰�

`GUI.java` �����û����棬���[GUI.java](src/main/java/helper/GUI.java)��

`img.java` ʵ����ͼ����صĲ������䷽�����£����[img.java](src/main/java/helper/img.java)��
```
public static BufferedImage readImage(String imageFile); /* ��ȡָ��·��ͼ��ΪBufferedImage����ͼ��*/

public static void writeImage(String imageFile, BufferedImage image); /* д��ָ��·��ͼ��ΪBufferedImage����ͼ��*/
 
public static BufferedImage array2Image(int[][][] rgb3DArray, int height, int width); /* ������rgb3DArrayת��ΪBufferedImage����ͼ��*/

public static int[][][] image2array(BufferedImage bf);/* ��BufferedImage����ͼ��ת��Ϊ����rgb3DArray*/
```

### Part 0c: org.example.unsupervised_machine_learning

`App.java` ������������ڣ����[App.java](src/main/java/org/example/unsupervised_machine_learning/App.java)��

## Part 1: K-means ����

k-means ������һ�ּ򵥵�ͼ��ָ���������������Ƶ����ݵ����һ��Ȼ����ƽ��ֵ�滻���ǵ�ֵ��

### Part 1a:`k_means` ���ʵ��

#### `get_initial_means()` 

�����������ѡ�� k ������Ϊ��Ⱥ�ĳ�ʼ�㣬��ʱ��Ⱥ�ľ�ֵ����ÿ�����ֵ��
```
public void get_initial_means() {
	NDArray newArr = shuffle(this.array, 0);                // �������
	this.mean = newArr.get(new NDIndex("0:" + k + ",:"));   // ѡ��ǰ k ����
	newArr.close();
}
```

#### `k_means_step()` 

���� k ����Ⱥ�ľ�ֵ�����ݽ��з��࣬ÿ���㶼���ֵ�һ����Ⱥ�У�֮�����ǽ����ݼ�Ⱥ�еĵ����¼��㼯Ⱥ�ľ�ֵ new_means��

```
public NDArray k_means_step(int m) {	
	// �����ķ�����
	NDArray clusters = manager.zeros(new Shape(m));
	int[][] axes = {{0},{1}};
	for(int i = 0; i < m; i++) {
		NDArray curSample = array.get(i);  // ��ǰ������
			
		// ���������Ķ�����(�����������Ⱥ����ֵ�ľ���)
		NDArray diff = mean.sub(curSample);
		NDArray norm = diff.norm(axes[1]);
			
		// ���࣬ÿ���������ൽ����������ļ�Ⱥ
		NDArray cluster = norm.argMin();
		clusters.set(new NDIndex(i), cluster);
			
		// ����
		diff.close();
		curSample.close();
		norm.close();
		cluster.close();
	}
		
	// ��������������
	NDArray index = manager.arange(m);
	// ����ÿ����Ⱥ���µ�����ֵ
	for(int i = 0; i < k; i++) {
		// ���ڼ�Ⱥ i ����������������
		NDArray mask = clusters.eq(i);
		NDArray cur_index = index.get(mask);
		// ���ڼ�Ⱥ i ����������
		NDArray cluster_samples = array.get(cur_index);
			
		// ��Ⱥ����ֵ���������ڸü�Ⱥ�������ľ�ֵ
		if(!cluster_samples.isEmpty()) {
			NDArray mean = cluster_samples.mean(axes[0]);
			this.mean.set(new NDIndex(i + ",:"), mean);
			mean.close();
		}
			
		// ����
		mask.close();
		cur_index.close();
		cluster_samples.close();
	}
	index.close();
	return clusters;
}
```

#### `segment()` 

ʹ�� K-means �㷨��ͼ��� RGB ֵ���뵽 k ����Ⱥ�У�Ȼ��ͼ���ԭʼֵ�滻Ϊ��Ӧ�ļ�Ⱥ����ֵ���ع�ͼ���Թ۲�Ч����

��Ȼ��������ľ���ֹͣ�仯ʱ���㷨�ﵽ������Ȼ������������Ĺ��̿����൱������������趨һ��������������

```
public BufferedImage segment(int max_iter) {
	// ԭͼ�Ĵ�С
	int r = (int)shape.getShape()[0];
	int c = (int)shape.getShape()[1];
	int ch = (int)shape.getShape()[2];
	int m = r*c;
		
	// ����һ�� k_means_step ���Դ���ľ���
	Shape arr_shape = new Shape(m, ch);
	this.array = this.o_array.reshape(arr_shape);
		
	// ��ʼ��
	get_initial_means();
		
	NDArray preClusters = manager.zeros(new Shape(m));     // ԭ��Ⱥ  
	// ����ֱ��������ﵽ����������
	for(int i = 0; i < max_iter; i++) {
		System.out.println(i);
		NDArray curClusters = k_means_step(m); // �µļ�Ⱥ
			
		// ����ʱ��������
		if(preClusters.contentEquals(curClusters)) {
			break;
		}
			
		preClusters.set(new NDIndex(":"), curClusters);
		curClusters.close();
	}
		
	// ��ԭʼֵ�滻Ϊ��Ӧ�ļ�Ⱥֵ���ع�ͼ��
	this.array.close();
	NDArray updated_array = mean.get(preClusters);
	this.array = updated_array.reshape(shape).toType(DataType.INT32, false);
	int[][][] rgb3DArray = get_rgbarray(r, c);
	BufferedImage update_Bufimg = img.array2Image(rgb3DArray, r, c);
	preClusters.close();
	updated_array.close();
	return update_Bufimg;
}
```

### Part 1b:����Ч��

���ӻ� K-means ����ͼ��ָ�Ľ��

�趨��ȺֵΪ5������������Ϊ32

![](resource/k_means_output_5.jpg)

�趨��ȺֵΪ16������������Ϊ32

![](resource/k_means_ouput_16.jpg)

### Part 1c:jar ������

��Ҫ�� pom.xml �ļ����������²����
```
<groupId>org.springframework.boot</groupId>
<artifactId>spring-boot-maven-plugin</artifactId>
<!-- maven���̴�������Խ��ֶ�����İ�Ҳ���ý�Ŀ��jar�� -->

<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-compiler-plugin</artifactId>
<!-- ָ����ĿԴ��� jdk �汾�������� jdk �汾 -->

<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-assembly-plugin</artifactId>
<!-- ����Ŀ��������������ģ�顢վ���ĵ����������ļ�һ����װ��һ���ɷַ��Ĺ鵵�ļ���-->

<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-jar-plugin</artifactId>
<!-- �趨 MAINFEST .MF�ļ��Ĳ���������ָ�����е�Main class����������jar������classpath�еȵ�-->
```

ʹ�� Maven Install ���д����

����λ�� target �ļ����µ���������k_means.jar �� k_means-jar-with-dependencies.jar��

�������о��趨����������ڣ��������С�

�������ڣ�k_means-jar-with-dependencies.jar ��������������������ֱ�����С�

## Part 2:  ʵ��һ����Ԫ��˹���ģ��

���������������������˹�ֲ���ʵ��һ����Ԫ��˹�������(Multivariate Gaussian Expectation Maximization)�㷨��EM �㷨��һ��ͨ���㷨�����������ǵ�������ͳ�Ʒֲ��������Ȼ�������������ǵķ����У����ǹ�ע���Ǹ�˹���ģ�͵������Ȼ������(MLEs)��

- - - 

���º������ᱻ��װ��һ����Ԫ��˹��������㷨��

1. ����������ݵ�(�������ص� rgb ֵ)�����ض���˹�����ĸ��ʡ�

2. ʹ���������(EM)ѵ��ģ�ͣ���ͼ���ʾΪ��˹�Ļ�ϡ�

3. ����ѵ��ģ�͵Ķ�����Ȼ���� default_convergence() ��Ϊ����������������(�μ� helper_functions.py �ļ�)��������� 10�ε������µ���Ȼֵ��֮ǰ��Ȼֵ�� 10% ���ڣ���ģ���Ѿ�������

4. ����ѵ���õ�ģ�Ͷ�ͼ����зָ

5. ͨ����ģ��ѵ���������ϵ�����ȷ����ѷָ��Ϊ EM ���ܱ�֤������ȫ�����ֵ����ʼ��ʱ k �����ѡȡ��ȫ������ģ�ʵ�����㷨�����ʼ���� k �������Ƿǳ����еġ�

#### ��Ҫ��ɵĺ�����
1. `initialize_parameters()`
2. `prob()`
3. `E_step()`
4. `M_step()`
5. `likelihood()`
6. `train_model()`
7. `cluster()`
8. `segment()`
9. `best_segment()`

## Part 3:  ģ�͸Ľ�

���ڣ����ǽ����������Ľ�����߸�˹���ģ�͵����ܡ�

### Part 3a:  �Ľ���ʼ������

ͨ��ѵ��һ����˹���ģ����Ѱ�ҳ�ʼ��ֵ���������͵ĳ�ʼ����ͬ�ڼ򵥵�����ѵ��ʱ�䣬��Ϊ���ǡ����á���Э����ͻ��ϵ��������

Ҳ����˵������ѵ�������Ǹ��ݳ�ʼ���ڼ�ѧ���ľ�ֵ���¼���Э������������ٴν����ϵ������Ϊ���ȷֲ���

һ�� GMM ������������������Э�������ͨ��������Щ���������и��ߵĻ������ֲ�����ֵ������������ʼ������

### Part 3b: ��������

ʵ�� `new_convergence_condition()` ������������һ���µ������������������ 10�ε����������µ�ģ�Ͳ���(��ֵ������ͻ��ϵ��)����ǰһ�������� 10%���ڣ��򷵻� true��

ͬʱ��������Ҫ�� `train_model_improved()` ����������ʵ�� `train_model()` ��

## Part 4: ��Ҷ˹��Ϣ׼��

�����������У�����ѡ��ģ�͵�Ψһ��׼�����Ƿ�ʹ������Ȼ��󻯣�����������Ҫ���ٲ�������ˣ�����ѡ�ģ�Ϳ���ֻ�Ǿ�����������ģ�ͣ��⽫��ѵ�����ݹ�����ϡ�

Ϊ�˱������ϣ����ǿ���ʹ�ñ�Ҷ˹��Ϣ׼��(BIC)���������ģ��ʹ�õĲ����������ͷ�ģ�͡�����ͷ��������Ӷȳͷ��;��ȳͷ�����Ȼ�����Ӷȳͷ���ģ�Ͳ���������Ҫ�Ǽ�Ⱥ�����йأ����ȳͷ�����Ȼֵ�йء�

### Part 4a: ʵ�ֱ�Ҷ˹��Ϣ׼��

ʵ�� `bayes_info_criterion()` �������������ڼ���ѵ���õĸ�˹���ģ�͵� BIC��

BIC �ļ��㹫ʽ���£�
BIC = ln(m)k�C2 * In(L) (m - ��������, k - ������, L - ��Ȼֵ)  
ֵ��ע����ǣ�`k` �� BIC ����ģ�͹��ƵĲ��������������Ǿ���ֵ����Ȼ��`k` �ɾ�ֵ��Э����ͻ��ϵ���Ĳ�������ȷ����

### Part 4b: ���Ա�Ҷ˹��Ϣ׼��

�� `BIC_likelihood_model_test()`�����У��㽫ʹ�� BIC �� ��Ȼֵ ��ȷ�� `image_matrix` ����Ѿ���ֵ��

ʹ�� `train_model_improved()` ������ �������ṩ�ľ�ֵ�б�`comp_means`���� ѵ��һ����С�� BIC ��ģ�ͺ�һ�������Ȼֵ��ģ�͡�

`comp_means` ��һ���б�����ÿ��Ԫ����һ�� k x n �ľ�ֵ����k �Ǿ���ֵ����

## ���ڵ�����

### �ڴ�й©

NDArray ����ʹ���ٱ����ã�Ҳ���ᱻ JVM ��Ϊ�ڴ��������д���������ʹ����ʹ�� `close()` ��������Ч�ػ��ձ�ռ�õ��ڴ档

��Ȼ NDmanger ʵ�� AutoCloseable �� close �ӿڣ������ resources ������ NDArray �� close ���������Ǳ���Ŀ��������Ȼ��ȡ��������Ĳ��ԡ�

Ȼ�����Ⲣû����ȫ������⡣

�����Ѿ��ͷ��˾����ܶ���ڴ棬������������ռ�õ��ڴ���Ȼ�ڻ���������

���ƺ��� DJL ����� bug�� NDManager ��ʹִ���� close�����ǻ��Ǳ����ã����������޷������ա�

Ŀǰ���޽��������

