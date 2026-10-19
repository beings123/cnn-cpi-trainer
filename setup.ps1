<#
.SYNOPSIS
一键创建虚拟环境并安装依赖
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    创建 Python 虚拟环境" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$VenvDir = "venv"
$PythonCmd = "python"

# 检查Python
try {
    $pyVersion = & $PythonCmd --version 2>&1
    Write-Host "[OK] 检测到 Python: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "[错误] 未找到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    pause
    exit 1
}

# 创建虚拟环境
if (Test-Path "$VenvDir\Scripts\python.exe") {
    Write-Host "[跳过] 虚拟环境已存在" -ForegroundColor Yellow
} else {
    Write-Host "[等待] 创建虚拟环境中..." -ForegroundColor Yellow
    & $PythonCmd -m venv $VenvDir
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 虚拟环境创建成功" -ForegroundColor Green
    } else {
        Write-Host "[错误] 虚拟环境创建失败" -ForegroundColor Red
        pause
        exit 1
    }
}

Write-Host ""
Write-Host "[等待] 安装依赖..." -ForegroundColor Yellow
Write-Host ""

$PyExe = "$VenvDir\Scripts\python.exe"
$PipExe = "$VenvDir\Scripts\pip.exe"

# 升级pip
& $PyExe -m pip install --upgrade pip -q | Out-Null

# 安装依赖（只有numpy和pandas，超轻量！）
& $PipExe install numpy pandas -q

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] 依赖安装完成！" -ForegroundColor Green
    Write-Host "     - numpy" 
    Write-Host "     - pandas"
} else {
    Write-Host "[错误] 依赖安装失败" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    环境就绪！双击 开始训练.bat 启动训练" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
pause
