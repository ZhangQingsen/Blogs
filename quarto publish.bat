@echo off
REM 设置UTF-8编码
chcp 65001 > nul

REM 设置文本颜色
color 70

REM 激活虚拟环境
call .venv\Scripts\activate

REM 执行quarto命令
quarto render
if %ERRORLEVEL% NEQ 0 (
    echo Quarto render failed.
    goto :eof
)

quarto publish gh-pages
if %ERRORLEVEL% NEQ 0 (
    echo Quarto publish failed.
    goto :eof
)

REM 显示完成效果
echo.
echo Quarto render and publish completed successfully.
echo Press any key to exit...
pause > nul

REM 退出虚拟环境
call deactivate
