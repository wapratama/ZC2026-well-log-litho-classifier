<#
.SYNOPSIS
    Deploy the lithology API to Google Cloud Run using the image you already built and tested locally.

.EXAMPLE
    docker build -t lithology-api .
    .\deploy\cloud_run.ps1 -ProjectId my-gcp-project

.NOTES
    Prerequisites: Docker Desktop running, the gcloud CLI installed, `gcloud auth login` done,
    and billing enabled on the GCP project (Cloud Run has a monthly free tier).
    The service is public (--allow-unauthenticated). Delete it when you no longer need it:
        gcloud run services delete lithology-api --region asia-southeast2
#>
param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [string]$Region = "asia-southeast2",        # Jakarta
    [string]$Repository = "lithology",
    [string]$Service = "lithology-api",
    [string]$LocalImage = "lithology-api",
    [string]$Tag = "v1"
)

$ErrorActionPreference = "Stop"
$Registry = "${Region}-docker.pkg.dev"
$Image = "${Registry}/${ProjectId}/${Repository}/${Service}:${Tag}"

function Invoke-Step {
    param([string]$Description, [scriptblock]$Command)
    Write-Host "==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Description (exit code $LASTEXITCODE)" }
}

Invoke-Step "Select project $ProjectId" { gcloud config set project $ProjectId }
Invoke-Step "Enable Cloud Run and Artifact Registry APIs" {
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com
}

# Create the Docker repository only if it does not exist yet.
$previous = $ErrorActionPreference
$ErrorActionPreference = "Continue"
gcloud artifacts repositories describe $Repository --location $Region *> $null
$repoExists = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $previous
if (-not $repoExists) {
    Invoke-Step "Create Artifact Registry repository $Repository" {
        gcloud artifacts repositories create $Repository --repository-format=docker `
            --location=$Region --description="Lithology API images"
    }
}

Invoke-Step "Let Docker authenticate to $Registry" { gcloud auth configure-docker $Registry --quiet }
Invoke-Step "Tag image as $Image" { docker tag $LocalImage $Image }
Invoke-Step "Push image" { docker push $Image }

Invoke-Step "Deploy to Cloud Run" {
    gcloud run deploy $Service `
        --image $Image `
        --region $Region `
        --port 8000 `
        --memory 1Gi `
        --cpu 1 `
        --cpu-boost `
        --min-instances 0 `
        --max-instances 2 `
        --allow-unauthenticated
}

$Url = gcloud run services describe $Service --region $Region --format "value(status.url)"
Write-Host "`nService URL: $Url" -ForegroundColor Green
Write-Host "Health check:" -ForegroundColor Green
Invoke-RestMethod "$Url/health" | ConvertTo-Json
Write-Host "`nTry it:  uv run python example_request.py --url $Url --well 16/2-7 --start 9000 --rows 600"
