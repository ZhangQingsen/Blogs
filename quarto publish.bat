@echo off
REM 设置UTF-8编码
chcp 65001 > nul

REM 设置文本颜色
color 70

REM 激活虚拟环境
call .venv\Scripts\activate

@REM REM 执行quarto命令
@REM quarto render
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo Quarto render failed.
@REM     goto :eof
@REM )

@REM quarto publish gh-pages --no-render
@REM if %ERRORLEVEL% NEQ 0 (
@REM     echo Quarto publish failed.
@REM     echo Press any key to exit...
@REM     pause > nul
@REM     goto :eof
@REM )

REM 显示完成效果
echo.
echo Quarto render and publish completed successfully.
echo Press any key to exit...
pause > nul

REM 退出虚拟环境
call deactivate
