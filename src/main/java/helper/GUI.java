package helper;


import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.*;

import java.awt.image.BufferedImage;

import EM_model.*;


public class GUI {
	JFrame f;
	JLabel origin_img, update_img;
	JLabel origin_text, update_text;
	Container c;
	JPanel origin, update, buttons;
	JButton select, train;
	FileDialog open;
	BufferedImage origin_Bufimg, update_Bufimg;
	JTextArea K;
	Font font = new Font("仿宋", Font.PLAIN, 35);
	Dimension preferredSize = new Dimension(100,40);
	public GUI() {
		f = new JFrame("k-means");
		c = f.getContentPane();
		c.setLayout(new FlowLayout(FlowLayout.CENTER, 0, 0));
		
		// 三个面板，分别放原图、重构图和按钮
		origin = new JPanel();
		origin.setBackground(Color.GREEN);
		origin.setPreferredSize(new Dimension(600, 300));
		update = new JPanel();
		update.setBackground(Color.YELLOW);
		update.setPreferredSize(new Dimension(600, 300));
		buttons = new JPanel();
		buttons.setBackground(Color.lightGray);
		buttons.setPreferredSize(new Dimension(600, 100));
		
		// 原图
		origin_text = new JLabel("-----------原图-----------");
		origin_text.setFont(font);
		origin_img = new JLabel();
		origin.setLayout(new FlowLayout(FlowLayout.CENTER, 10, 10));
		origin.add(origin_text);
		origin.add(origin_img);
		
		// 重构图
		update_text = new JLabel("-----------重构图-----------");
		update_text.setFont(font);
		update_img = new JLabel();
		update.setLayout(new FlowLayout(FlowLayout.CENTER, 10, 10));
		update.add(update_text);
		update.add(update_img);
		
		// 创建文本对话框用于选择图像
		open = new FileDialog(f, "选择需要加载的文件", FileDialog.LOAD);
		
		// 按钮
		select = new JButton("选择图像");
		select.setPreferredSize(preferredSize);
		JLabel cluster_num= new JLabel("集群数");
		K = new JTextArea(1,5);
		train = new JButton("图像聚类");
		train.setPreferredSize(preferredSize);
		button_set();
		buttons.setLayout(new FlowLayout(FlowLayout.CENTER, 10, 10));
		buttons.add(select);
		buttons.add(cluster_num);
		buttons.add(K);
		buttons.add(train);
		
		f.add(origin);
		f.add(update);
		f.add(buttons);
		
		f.setBounds(200, 200, 600, 700);
		f.setVisible(true);
		f.setResizable(false);
		f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
	
	public void button_set() {
		// select打开文本对话框
		select.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                open.setVisible(true);
                // 用户选择的文件路径和名称
                String imageFile = open.getDirectory() + open.getFile();
                // 显示图像并将其转化为BufferedImage
                set_origin_img(imageFile);
            }
        });
		
		// train进行聚类
		train.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {    	
            	// 创建模型进行聚类
            	int k = Integer.parseInt(K.getText());
            	k_means model = new k_means(origin_Bufimg, k);
            	update_Bufimg = model.segment(32);
            	model.close();
            	
            	// 显示经过聚类后重构的图像
            	update_img.setIcon(new ImageIcon(update_Bufimg));
            }
        });
	}
	
	public void set_origin_img(String imageFile) {
		origin_Bufimg = img.readImage(imageFile);
		// 设定显示的原图
		origin_img.setIcon(new ImageIcon(origin_Bufimg));
	}
}
