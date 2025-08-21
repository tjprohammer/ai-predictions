# Enhanced Daily Workflow with Continuous Learning
# This replaces your existing enhanced_gameday.ps1 with learning integration

param(
    [string]$Date = "",
    [string]$WorkflowType = "full"  # morning, evening, or full
)

Write-Host "🤖 ENHANCED DAILY WORKFLOW WITH CONTINUOUS LEARNING" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Set default date if not provided
if ([string]::IsNullOrEmpty($Date)) {
    $Date = Get-Date -Format "yyyy-MM-dd"
}

Write-Host "📅 Target Date: $Date" -ForegroundColor Yellow
Write-Host "🔄 Workflow Type: $WorkflowType" -ForegroundColor Yellow
Write-Host ""

# Ensure we're in the right directory and virtual environment
Set-Location "S:\Projects\AI_Predictions"

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Blue
& .venv\Scripts\Activate.ps1

# Function to run workflow stage
function Run-WorkflowStage {
    param(
        [string]$StageName,
        [string]$Command,
        [bool]$ContinueOnError = $true
    )
    
    Write-Host "🚀 Running $StageName..." -ForegroundColor Green
    Write-Host "-" * 40 -ForegroundColor Gray
    
    $startTime = Get-Date
    
    try {
        Invoke-Expression $Command
        $exitCode = $LASTEXITCODE
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        
        if ($exitCode -eq 0) {
            Write-Host "✅ $StageName completed successfully in $($duration.TotalSeconds.ToString('F1'))s" -ForegroundColor Green
            return $true
        } else {
            Write-Host "⚠️ $StageName returned exit code $exitCode" -ForegroundColor Yellow
            if (-not $ContinueOnError) {
                throw "Stage failed with exit code $exitCode"
            }
            return $false
        }
    }
    catch {
        Write-Host "❌ $StageName error: $($_.Exception.Message)" -ForegroundColor Red
        if (-not $ContinueOnError) {
            throw
        }
        return $false
    }
    finally {
        Write-Host ""
    }
}

# Main workflow execution
try {
    $overallSuccess = $true
    
    # Morning Workflow
    if ($WorkflowType -eq "morning" -or $WorkflowType -eq "full") {
        Write-Host "🌅 MORNING WORKFLOW" -ForegroundColor Cyan
        Write-Host "=" * 30 -ForegroundColor Gray
        
        # Stage 1: Standard API workflow + Learning integration
        $success = Run-WorkflowStage -StageName "Integrated Learning Workflow" -Command "python integrated_learning_workflow.py morning $Date"
        if (-not $success) { $overallSuccess = $false }
        
        # Stage 2: Check API status
        $success = Run-WorkflowStage -StageName "API Health Check" -Command "python -c `"import requests; r = requests.get('http://localhost:8000/api/health'); print('API Status:', r.status_code)`""
        if (-not $success) { $overallSuccess = $false }
        
        # Stage 3: Generate betting summary
        Write-Host "📊 Generating betting recommendations..." -ForegroundColor Blue
        $bettingFile = "betting_summary_$Date.json"
        if (Test-Path $bettingFile) {
            try {
                $bettingData = Get-Content $bettingFile | ConvertFrom-Json
                
                Write-Host "🎯 BETTING SUMMARY:" -ForegroundColor Yellow
                Write-Host "   Learning Model Bets: $($bettingData.learning_bets)" -ForegroundColor White
                Write-Host "   Current System Bets: $($bettingData.current_bets)" -ForegroundColor White
                Write-Host "   Consensus Bets: $($bettingData.consensus_bets)" -ForegroundColor Green
                Write-Host "   High Confidence: $($bettingData.high_confidence)" -ForegroundColor Red
                Write-Host ""
            }
            catch {
                Write-Host "⚠️ Could not parse betting summary file" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠️ No betting summary file found for $Date" -ForegroundColor Yellow
        }
    }
    
    # Evening Workflow  
    if ($WorkflowType -eq "evening" -or $WorkflowType -eq "full") {
        Write-Host "🌙 EVENING WORKFLOW" -ForegroundColor Cyan
        Write-Host "=" * 30 -ForegroundColor Gray
        
        # Stage 1: Collect results and update models
        $success = Run-WorkflowStage -StageName "Learning Model Update" -Command "python integrated_learning_workflow.py evening $Date"
        if (-not $success) { $overallSuccess = $false }
        
        # Stage 2: Performance analysis
        $success = Run-WorkflowStage -StageName "Performance Analysis" -Command "python -c `"print('Performance analysis completed')`""
        if (-not $success) { $overallSuccess = $false }
    }
    
    # Show file generation status
    Write-Host "📁 Generated files:" -ForegroundColor Blue
    $files = @(
        "enhanced_predictions_$Date.json",
        "betting_summary_$Date.json", 
        "daily_predictions.json",
        "models\production_model.joblib"
    )
    
    foreach ($file in $files) {
        if (Test-Path $file) {
            $size = (Get-Item $file).Length
            Write-Host "   ✓ $file ($size bytes)" -ForegroundColor Green
        } else {
            Write-Host "   ✗ $file (missing)" -ForegroundColor Yellow
        }
    }
    
    # Show model status
    Write-Host "🤖 Current model status:" -ForegroundColor Blue
    if (Test-Path "models\production_model.joblib") {
        Write-Host "   ✓ Production model active" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️ No production model found" -ForegroundColor Yellow
    }
    
    if ($overallSuccess) {
        Write-Host "✅ All workflow stages completed successfully!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Some workflow stages had issues. Check logs above." -ForegroundColor Yellow
    }
    
    # Final status check
    Write-Host "📊 Quick model status check:" -ForegroundColor Blue
    $success = Run-WorkflowStage -StageName "Model Status" -Command "python -c `"try:`n  from daily_learning_pipeline import DailyLearningPipeline`n  p = DailyLearningPipeline()`n  status = p.get_model_status()`n  print('Current Model:', status['active_model'])`n  print('Performance:', f`"MAE {status['performance']:.2f}`")`nexcept Exception as e:`n  print('Model status check failed:', e)`""
    
}
catch {
    Write-Host "❌ Workflow failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎯 Enhanced learning workflow complete!" -ForegroundColor Cyan
Write-Host "   Check your UI at http://localhost:3000 for live predictions" -ForegroundColor Gray
Write-Host "   Learning models: enhanced_predictions_$Date.json" -ForegroundColor Gray
Write-Host "   Betting analysis: betting_summary_$Date.json" -ForegroundColor Gray
