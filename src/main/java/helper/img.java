package helper;


import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

import java.awt.Color;
import java.awt.image.BufferedImage;


public class img {
	/* 读取指定路径图像为BufferedImage类型图像*/
    public static BufferedImage readImage(String imageFile){
        File file = new File(imageFile);
        BufferedImage bf = null;
        try {
            bf = ImageIO.read(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bf;
    }
    
    /* 写入指定路径图像为BufferedImage类型图像*/
    public static void writeImage(String imageFile, BufferedImage image){
    	File ImageFile = new File(imageFile);
        try {
        	ImageIO.write(image, "png", ImageFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    /* 将数组rgb3DArray转换为BufferedImage类型图像*/
    public static BufferedImage array2Image(int[][][] rgb3DArray, int height, int width){
        int r = 0;
        int g = 1;
        int b = 2;
    	BufferedImage image = new BufferedImage(width, height,  
    			BufferedImage.TYPE_INT_RGB);
    	
    	for(int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
            	int rvalue = rgb3DArray[i][j][r];
            	int gvalue = rgb3DArray[i][j][g];
            	int bvalue = rgb3DArray[i][j][b];
            	int rgbvalue = new Color(rvalue, gvalue, bvalue).getRGB();
            	image.setRGB(j, i, rgbvalue);
            }
        }
        return image;
    }

    /* 将BufferedImage类型图像转换为数组rgb3DArray*/
    public static int[][][] image2array(BufferedImage bf) {
        // 获取图片宽度和高度
        int width = bf.getWidth();   // 图片宽度
        int height = bf.getHeight();  //图片高度
        int r = 0;
        int g = 1;
        int b = 2;
        int[] data = new int[width*height];
        bf.getRGB(0, 0, width, height, data, 0, width);
        
        int[][][] rgb3DArray = new int[height][width][3];
        // 将二维数组转换为三维数组（宽度*高度*通道数）
        for(int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                rgb3DArray[i][j][r] = (data[i*width + j] & 0xff0000) >> 16;
                rgb3DArray[i][j][g] =  (data[i*width + j] & 0xff00) >> 8;
                rgb3DArray[i][j][b] =  (data[i*width + j] & 0xff);
            }
        }
        return rgb3DArray;
    }
}
