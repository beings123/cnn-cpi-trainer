<#
.SYNOPSIS
一键运行 CNN 共同富裕预测训练
.PARAMETER Light
极简模式：更少的epoch，更快完成
.PARAMETER Epochs
自定义训练轮数
#>

param(
    [switch]$Light,
    [int]$Epochs = 80
)

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    CNN 共同富裕指数预测训练器" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$PyExe = "venv\Scripts\python.exe"

# 检查虚拟环境
if (-not (Test-Path $PyExe)) {
    Write-Host "[警告] 虚拟环境不存在，先执行 setup.ps1 创建环境..." -ForegroundColor Yellow
    & .\setup.ps1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 环境创建失败" -ForegroundColor Red
        pause
        exit 1
    }
}

# 检查数据
if (-not (Test-Path "data\common_prosperity_data.csv")) {
    Write-Host "[信息] 生成训练数据中..." -ForegroundColor Yellow
    & $PyExe generate_data.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[错误] 数据生成失败" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host ""
}

# 构建参数
$trainArgs = @()
if ($Light) {
    $trainArgs += "--light"
} else {
    $trainArgs += "--epochs"
    $trainArgs += $Epochs.ToString()
}

# 运行训练
Write-Host "[启动] 开始训练（按 Ctrl+C 可中断，自动保存最优模型）" -ForegroundColor Green
Write-Host ""

& $PyExe train_cnn.py @trainArgs

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] 训练完成！最优模型保存在 models\best_cnn_model.h5" -ForegroundColor Green
} else {
    Write-Host "[警告] 训练中断或出错" -ForegroundColor Yellow
}
Write-Host ""
pause
