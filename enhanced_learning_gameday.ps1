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
            Write-Host "❌ $StageName failed with exit code $exitCode" -ForegroundColor Red
            if (-not $ContinueOnError) {
                throw "$StageName failed"
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
        
        # Stage 1: Standard API workflow
        $success = Run-WorkflowStage -StageName "Standard API Workflow" -Command "python integrated_learning_workflow.py morning $Date"
        if (-not $success) { $overallSuccess = $false }
        
        # Stage 2: Check API status
        $success = Run-WorkflowStage -StageName "API Health Check" -Command "python test_enhanced_api.py"
        if (-not $success) { $overallSuccess = $false }
        
        # Stage 3: Generate betting summary
        Write-Host "📊 Generating betting recommendations..." -ForegroundColor Blue
        $bettingFile = "betting_summary_$Date.json"
        if (Test-Path $bettingFile) {
            $bettingData = Get-Content $bettingFile | ConvertFrom-Json
            
            Write-Host "🎯 BETTING SUMMARY:" -ForegroundColor Yellow
            Write-Host "   Learning Model Bets: $($bettingData.learning_bets)" -ForegroundColor White
            Write-Host "   Current System Bets: $($bettingData.current_bets)" -ForegroundColor White
            Write-Host "   Consensus Bets: $($bettingData.consensus_bets.Count)" -ForegroundColor Green
            Write-Host "   High Confidence: $($bettingData.high_confidence_learning.Count)" -ForegroundColor Red
            Write-Host ""
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
        $success = Run-WorkflowStage -StageName "Performance Analysis" -Command "python simple_results_checker.py"
        if (-not $success) { $overallSuccess = $false }
        
        # Stage 3: Weekly summary (if it's end of week)
        $dayOfWeek = (Get-Date $Date).DayOfWeek
        if ($dayOfWeek -eq "Sunday") {
            $success = Run-WorkflowStage -StageName "Weekly Performance Summary" -Command "python weekly_performance_tracker.py"
            if (-not $success) { $overallSuccess = $false }
        }
    }
    
    # Full day analysis
    if ($WorkflowType -eq "full") {
        Write-Host "📈 FULL DAY ANALYSIS" -ForegroundColor Cyan
        Write-Host "=" * 30 -ForegroundColor Gray
        
        # Historical comparison
        $success = Run-WorkflowStage -StageName "Historical Performance Check" -Command "python historical_performance_analysis.py"
        if (-not $success) { $overallSuccess = $false }
    }
    
    # Final summary
    Write-Host "🏁 WORKFLOW SUMMARY" -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Gray
    
    if ($overallSuccess) {
        Write-Host "✅ All workflow stages completed successfully!" -ForegroundColor Green
        
        # Show key files generated
        Write-Host "📁 Generated files:" -ForegroundColor Blue
        
        $files = @(
            "enhanced_predictions_$Date.json",
            "betting_summary_$Date.json",
            "learning_predictions_$Date.json"
        )
        
        foreach ($file in $files) {
            if (Test-Path $file) {
                $size = (Get-Item $file).Length
                Write-Host "   ✓ $file ($size bytes)" -ForegroundColor Green
            }
        }
        
        # Show model status
        Write-Host "🤖 Current model status:" -ForegroundColor Blue
        if (Test-Path "models\production_model.joblib") {
            Write-Host "   ✓ Production model active" -ForegroundColor Green
        } else {
            Write-Host "   ⚠️ No production model found" -ForegroundColor Yellow
        }
        
    } else {
        Write-Host "⚠️ Some workflow stages had issues. Check logs above." -ForegroundColor Yellow
    }
    
    # Quick performance check
    Write-Host "📊 Quick performance check:" -ForegroundColor Blue
    $success = Run-WorkflowStage -StageName "Model Status Check" -Command "python -c `"from daily_learning_pipeline import DailyLearningPipeline; p = DailyLearningPipeline(); print('Model Status:', p.get_model_status())`""
    
} catch {
    Write-Host "❌ Workflow failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "🎯 Daily workflow complete! Check the UI at http://localhost:3000 for live predictions." -ForegroundColor Cyan
