---
title: "VSCode用于cpp调试的相关设置"
description: 妈妈再也不用担心我没空间装Visual Studio啦~
toc: false
layout: post
categories: [vscode, cpp]
author: BjChacha
---

## 0. 前言

由于对Cpp的需求比较轻量，用来学习语法和写写算法题之类的，用`visual studio`觉得十分臃肿。`vscode`是一款非常优秀的代码编辑器，本身不具备任何编译器等IDE功能。需要进行一定的环境准备才能用`vscode`进行Cpp代码的编译和调试。

## 1. GCC编译器安装
 
1.下载[MinGW-w64](https://sourceforge.net/projects/mingw-w64/files/)，选择最新版本的**x86_64-posix-seh**。    
2.解压下载下来的压缩包，并将`mingw64`文件夹移动到C盘根目录下（当然也可以到其他地方）。  
3.将`mingw64`文件夹下的`bin`文件夹目录添加到用户环境变量中的`PATH`中。此例中则将`C:\mingw64\bin`添加到PATH中。  
4.验证gcc编译器安装是否成功：**Win+R**运行cmd，输入gcc，若提示  

> gcc:fatal error: no input files

则gcc编译器安装成功；若提示：

> 不是内部命令或外部命令    

则gcc没有安装成功，回头检查环境变量中`PATH`中的目录与gcc实际目录是否符合。

## 2. vscode cpp相关插件

- C/C++：提供Cpp Debug等功能。

## 3. vscode 相关配置文件

在`vscode`打开Cpp工作区后，会在当前目录下自动生成`.vscode`文件夹。该文件夹下存放当前工作区的以下配置文件。若无以下文件可自行创建。

- *launch.json*   
  
        {
            "version": "0.2.0",
            "configurations": [
            
                {
                    "name": "(gdb) Launch",
                    "type": "cppdbg",
                    "request": "launch",
                    "program": "${fileDirname}/${fileBasenameNoExtension}.exe",
                    "args": [],
                    "stopAtEntry": false,
                    "cwd": "${workspaceFolder}",
                    "environment": [],
                    "externalConsole": true,
                    "internalConsoleOptions": "neverOpen",
                    "MIMode": "gdb",
                    "miDebuggerPath": "gdb.exe",
                    "setupCommands": [
                        {
                            "description": "Enable pretty-printing for gdb",
                            "text": "-enable-pretty-printing",
                            "ignoreFailures": false
                        }
                    ],
                    "preLaunchTask": "Compile",
                }
            ]
        }

- *tasks.json*

        {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Compile",
                    "command": "g++",
                    "args": [
                        "${file}",
                        "-o",
                        "${fileDirname}/${fileBasenameNoExtension}.exe",
                        "-g",
                        "-Wall",
                        "-static-libgcc",
                        "-fexec-charset=GBK"
                    ],
                    "type": "process",
                    "group": {
                        "kind": "build",
                        "isDefault": true
                    },
                    "presentation": {
                        "echo": true,
                        "reveal": "always",
                        "focus": false,
                        "panel": "shared",
                    },
                }
            ]
        }


- *setting.json* (部分)

        {
            ...
            "files.defaultLanguage": "cpp",
            "editor.formatOnType": true,
            "editor.suggest.snippetsPreventQuickSuggestions": false,
            "editor.acceptSuggestionOnEnter": "off",
            "editor.snippetSuggestions": "top",
            "code-runner.runInTerminal": true,
            "code-runner.executorMap": {
                "c": "cd $dir && gcc '$fileName' -o '$fileNameWithoutExt.exe' -Wall -g -O2 -static-libgcc -std=c11 -fexec-charset=GBK && &'$dir$fileNameWithoutExt'",
                "cpp": "cd $dir && g++ '$fileName' -o '$fileNameWithoutExt.exe' -Wall -g -O2 -static-libgcc -std=c++17 -fexec-charset=GBK && &'$dir$fileNameWithoutExt'"
            },
            "code-runner.saveFileBeforeRun": true,
            "code-runner.preserveFocus": true,
            "code-runner.clearPreviousOutput": false,
            "code-runner.ignoreSelection": true,
            "C_Cpp.clang_format_sortIncludes": true,
            ...
        }

## 4. 用vscode进行cpp调试
1.在任意目录下创建文件夹`cpp`。     
2.在`vscode`下`Ctrl+K`、`Ctrl+O`，打开`cpp`文件夹。   
3.`Ctrl+N`创建新文件，随便写个cpp demo:

    #include <iostream>
    using namespace std;

    int main()
    {
        cout << "hello world" << endl;
        int a;
        cin >> a;
        return 0;
    }

4.`Shift+Ctrl+B`进行编译；`F5`进行调试，会弹出命令行窗口显示`hello world`输出。至此完成所有设置。

## 参考
[知乎](https://www.zhihu.com/question/30315894)  
[vscode官方文档](https://code.visualstudio.com/docs/cpp/config-mingw)