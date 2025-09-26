# Multi-Region Deployment Script for Rap Analyzer
# Phase 2: Multi-Region Setup

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("deploy", "status", "failover", "cleanup")]
    [string]$Action,
    
    [ValidateSet("us-east-1", "us-west-2", "eu-west-1", "all")]
    [string]$Region = "all",
    
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host "🌍 Multi-Region Deployment Manager" -ForegroundColor Green
Write-Host "Action: $Action | Region: $Region" -ForegroundColor Yellow

# Configuration
$Regions = @{
    "us-east-1" = @{
        Name = "us-east-1-cluster"
        Zone = "primary"
        Context = "us-east-1"
        DbRole = "primary"
    }
    "us-west-2" = @{
        Name = "us-west-2-cluster" 
        Zone = "replica"
        Context = "us-west-2"
        DbRole = "replica"
    }
    "eu-west-1" = @{
        Name = "eu-west-1-cluster"
        Zone = "replica"
        Context = "eu-west-1"
        DbRole = "replica"
    }
}

function Test-Prerequisites {
    Write-Host "🔍 Checking prerequisites..." -ForegroundColor Cyan
    
    # Check kubectl
    try {
        kubectl version --client --short | Out-Null
        Write-Host "  ✅ kubectl found" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ kubectl not found" -ForegroundColor Red
        exit 1
    }
    
    # Check ArgoCD CLI
    try {
        argocd version --client --short | Out-Null
        Write-Host "  ✅ ArgoCD CLI found" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠️  ArgoCD CLI not found (optional)" -ForegroundColor Yellow
    }
    
    # Check Helm
    try {
        helm version --short | Out-Null
        Write-Host "  ✅ Helm found" -ForegroundColor Green
    } catch {
        Write-Host "  ❌ Helm not found" -ForegroundColor Red
        exit 1
    }
}

function Deploy-PostgreSQLPrimary {
    param([string]$RegionName)
    
    Write-Host "🐘 Deploying PostgreSQL Primary in $RegionName..." -ForegroundColor Magenta
    
    if ($DryRun) {
        Write-Host "  [DRY RUN] Would deploy PostgreSQL primary" -ForegroundColor Yellow
        return
    }
    
    # Switch to region context
    kubectl config use-context $Regions[$RegionName].Context
    
    # Apply PostgreSQL configurations
    kubectl apply -f multi-region/postgresql/postgresql-shared.yaml
    kubectl apply -f multi-region/postgresql/postgresql-primary.yaml
    
    # Wait for primary to be ready
    Write-Host "  ⏳ Waiting for PostgreSQL primary to be ready..." -ForegroundColor Yellow
    kubectl wait --for=condition=ready pod -l app=postgresql,role=primary -n rap-analyzer --timeout=300s
    
    Write-Host "  ✅ PostgreSQL primary deployed successfully" -ForegroundColor Green
}

function Deploy-PostgreSQLReplica {
    param([string]$RegionName)
    
    Write-Host "🐘 Deploying PostgreSQL Replica in $RegionName..." -ForegroundColor Magenta
    
    if ($DryRun) {
        Write-Host "  [DRY RUN] Would deploy PostgreSQL replica" -ForegroundColor Yellow
        return
    }
    
    # Switch to region context
    kubectl config use-context $Regions[$RegionName].Context
    
    # Apply shared configurations
    kubectl apply -f multi-region/postgresql/postgresql-shared.yaml
    
    # Apply replica configuration with region label
    $replicaConfig = Get-Content multi-region/postgresql/postgresql-replica.yaml | ForEach-Object {
        $_ -replace "region: us-west-2", "region: $RegionName"
    }
    $replicaConfig | kubectl apply -f -
    
    # Wait for replica to be ready
    Write-Host "  ⏳ Waiting for PostgreSQL replica to be ready..." -ForegroundColor Yellow
    kubectl wait --for=condition=ready pod -l app=postgresql,role=replica -n rap-analyzer --timeout=300s
    
    Write-Host "  ✅ PostgreSQL replica deployed successfully" -ForegroundColor Green
}

function Deploy-Application {
    param([string]$RegionName)
    
    $regionInfo = $Regions[$RegionName]
    Write-Host "🚀 Deploying Rap Analyzer Application in $RegionName..." -ForegroundColor Blue
    
    if ($DryRun) {
        Write-Host "  [DRY RUN] Would deploy application" -ForegroundColor Yellow
        return
    }
    
    # Switch to region context
    kubectl config use-context $regionInfo.Context
    
    # Deploy using Helm with region-specific values
    $valuesFile = "multi-region/clusters/values-$RegionName.yaml"
    
    helm upgrade --install rap-analyzer-$RegionName ./helm/rap-analyzer `
        --namespace rap-analyzer `
        --create-namespace `
        --values helm/rap-analyzer/values.yaml `
        --values $valuesFile `
        --set global.region=$RegionName `
        --set global.zone=$($regionInfo.Zone) `
        --wait --timeout=10m
    
    Write-Host "  ✅ Application deployed successfully in $RegionName" -ForegroundColor Green
}

function Deploy-MultiRegion {
    Write-Host "🌍 Starting Multi-Region Deployment..." -ForegroundColor Green
    
    # Deploy primary region first
    if ($Region -eq "all" -or $Region -eq "us-east-1") {
        Write-Host "`n📍 Deploying Primary Region (US East 1)..." -ForegroundColor Cyan
        Deploy-PostgreSQLPrimary -RegionName "us-east-1"
        Deploy-Application -RegionName "us-east-1"
    }
    
    # Wait for primary to be fully ready before deploying replicas
    if ($Region -eq "all") {
        Write-Host "`n⏳ Waiting 2 minutes for primary to stabilize..." -ForegroundColor Yellow
        if (-not $DryRun) { Start-Sleep -Seconds 120 }
    }
    
    # Deploy replica regions
    if ($Region -eq "all" -or $Region -eq "us-west-2") {
        Write-Host "`n📍 Deploying Replica Region (US West 2)..." -ForegroundColor Cyan
        Deploy-PostgreSQLReplica -RegionName "us-west-2"
        Deploy-Application -RegionName "us-west-2"
    }
    
    if ($Region -eq "all" -or $Region -eq "eu-west-1") {
        Write-Host "`n📍 Deploying Replica Region (EU West 1)..." -ForegroundColor Cyan
        Deploy-PostgreSQLReplica -RegionName "eu-west-1"
        Deploy-Application -RegionName "eu-west-1"
    }
    
    # Deploy ArgoCD ApplicationSet for ongoing management
    if ($Region -eq "all") {
        Write-Host "`n🔄 Deploying ArgoCD ApplicationSet..." -ForegroundColor Cyan
        if (-not $DryRun) {
            kubectl config use-context us-east-1
            kubectl apply -f multi-region/clusters/multi-region-clusters.yaml
        }
    }
    
    Write-Host "`n🎉 Multi-Region Deployment Complete!" -ForegroundColor Green
}

function Show-Status {
    Write-Host "📊 Multi-Region Status Report" -ForegroundColor Green
    
    foreach ($regionName in $Regions.Keys) {
        if ($Region -ne "all" -and $Region -ne $regionName) { continue }
        
        $regionInfo = $Regions[$regionName]
        Write-Host "`n📍 Region: $regionName ($($regionInfo.Zone))" -ForegroundColor Cyan
        
        try {
            kubectl config use-context $regionInfo.Context
            
            # PostgreSQL status
            Write-Host "  🐘 PostgreSQL:" -ForegroundColor Magenta
            $pgPods = kubectl get pods -l app=postgresql -n rap-analyzer --no-headers 2>$null
            if ($pgPods) {
                Write-Host "    $pgPods" -ForegroundColor Gray
            } else {
                Write-Host "    No PostgreSQL pods found" -ForegroundColor Yellow
            }
            
            # Application status
            Write-Host "  🚀 Application:" -ForegroundColor Blue
            $appPods = kubectl get pods -l app=rap-analyzer -n rap-analyzer --no-headers 2>$null
            if ($appPods) {
                Write-Host "    $appPods" -ForegroundColor Gray
            } else {
                Write-Host "    No application pods found" -ForegroundColor Yellow
            }
            
            # Service status
            Write-Host "  🌐 Services:" -ForegroundColor Green
            kubectl get svc -n rap-analyzer --no-headers 2>$null | ForEach-Object {
                Write-Host "    $_" -ForegroundColor Gray
            }
            
        } catch {
            Write-Host "  ❌ Unable to connect to $regionName cluster" -ForegroundColor Red
        }
    }
}

function Perform-Failover {
    Write-Host "🔄 Performing Failover Operation..." -ForegroundColor Yellow
    
    if (-not $Force) {
        $confirm = Read-Host "This will promote a replica to primary. Continue? (y/N)"
        if ($confirm -ne "y") {
            Write-Host "Failover cancelled" -ForegroundColor Yellow
            return
        }
    }
    
    # This would implement actual failover logic
    Write-Host "⚠️  Failover functionality requires additional implementation" -ForegroundColor Yellow
    Write-Host "  Manual steps:" -ForegroundColor Cyan
    Write-Host "  1. Stop primary PostgreSQL" -ForegroundColor Gray
    Write-Host "  2. Promote replica to primary" -ForegroundColor Gray
    Write-Host "  3. Update DNS/load balancer" -ForegroundColor Gray
    Write-Host "  4. Restart applications" -ForegroundColor Gray
}

function Cleanup-MultiRegion {
    Write-Host "🧹 Cleaning up Multi-Region Deployment..." -ForegroundColor Yellow
    
    if (-not $Force) {
        $confirm = Read-Host "This will delete all resources. Continue? (y/N)"
        if ($confirm -ne "y") {
            Write-Host "Cleanup cancelled" -ForegroundColor Yellow
            return
        }
    }
    
    foreach ($regionName in $Regions.Keys) {
        if ($Region -ne "all" -and $Region -ne $regionName) { continue }
        
        Write-Host "  🗑️  Cleaning up $regionName..." -ForegroundColor Cyan
        
        try {
            kubectl config use-context $Regions[$regionName].Context
            
            if (-not $DryRun) {
                helm uninstall rap-analyzer-$regionName -n rap-analyzer 2>$null
                kubectl delete namespace rap-analyzer 2>$null
            }
            
            Write-Host "    ✅ $regionName cleaned up" -ForegroundColor Green
            
        } catch {
            Write-Host "    ⚠️  Issues cleaning up $regionName" -ForegroundColor Yellow
        }
    }
}

# Main execution
try {
    Test-Prerequisites
    
    switch ($Action) {
        "deploy" { Deploy-MultiRegion }
        "status" { Show-Status }
        "failover" { Perform-Failover }
        "cleanup" { Cleanup-MultiRegion }
    }
    
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Operation completed successfully!" -ForegroundColor Green