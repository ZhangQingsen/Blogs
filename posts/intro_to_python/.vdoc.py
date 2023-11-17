# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
#| eval: false
@echo off
REM 声明采用UTF-8编码
chcp 65001
ECHO.
TITLE [环境安装]
color 70
@REM start/wait python-3.7.8-amd64.exe /passive PrependPath=1
@REM pause
call python -m venv venv
call dir
pause
call venv\Scripts\activate
call python -m pip install --upgrade pip
call pip3 --trusted-host pypi.tuna.tsinghua.edu.cn install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas
call pip3 --trusted-host pypi.tuna.tsinghua.edu.cn install -i https://pypi.tuna.tsinghua.edu.cn/simple jupyter
call pip3 list
call cmd

@REM pause
```
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
#| eval: false
@echo off
REM 声明采用UTF-8编码
chcp 65001
ECHO.
TITLE [环境安装]
color 70

call venv\Scripts\activate
echo [35mjupyter notebook stop[0m
call cmd


@REM pause
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
