---
title: 在WSL中安装MacOS虚拟机
description: 圆了我的苹果梦
toc: true
hide: false
layout: post
categories: [WSL]
author: BjChacha
---

## 0. 前言
在github上发现有大佬做了个[脚本](https://github.com/myspaghetti/macos-virtualbox)，可以**傻瓜式**创建MacOS虚拟机。一直很佩服这些做这种便捷易懂的教程的作者，真的希望自己也有能力做这些事情。

## 1. 环境
* [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) (Ubuntu)
* [VirtualBox](https://www.virtualbox.org/wiki/Downloads) 6.1.6 or higher **with Extension Pack**

## 2. 进入WSL
1. 打开Ubuntu终端，首先安装必要的依赖项: 
    ```
    sudo apt-get install coreutils gzip unzip xxd wget  
    sudo apt-get update -y  
    sudo apt-get install dmg2img
    ```

2. 下载大佬的[脚本](https://raw.githubusercontent.com/myspaghetti/macos-virtualbox/master/macos-guest-virtualbox.sh)，并在Ubuntu终端中运行。
   > WSL的默认路径是Windows下的用户文件夹，如C:\Users\username，cd几下就能定位到Windows的文件。  

   > 另外提示一下，建议硬盘可用容量50G左右。

3. 跟着大佬的脚本的提示一步步走。**一定要看脚本提示**，因为进入安装界面后还是需要脚本的自动化操作。期间虚拟机会重启几次，等待脚本完全结束后就可以自己操作了。

## 3. 凑数用