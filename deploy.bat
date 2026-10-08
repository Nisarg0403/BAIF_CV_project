@echo off
setlocal enabledelayedexpansion
echo ===================================================
echo   Deploying BAIF CattleWeightAI Update to Azure
echo ===================================================

REM Generate unique timestamp tag to bypass Azure image caching
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set dt=%%a
set BUILD_TAG=v-%dt:~0,8%-%dt:~8,6%

echo [1/4] Logging into Azure Container Registry...
call az acr login --name acrcattleweightai

echo [2/4] Building Docker container (Tag: %BUILD_TAG%)...
docker build -t acrcattleweightai.azurecr.io/baif-cattleweight-app:latest -t acrcattleweightai.azurecr.io/baif-cattleweight-app:%BUILD_TAG% .

echo [3/4] Pushing container images to ACR...
docker push acrcattleweightai.azurecr.io/baif-cattleweight-app:latest
docker push acrcattleweightai.azurecr.io/baif-cattleweight-app:%BUILD_TAG%

echo [4/4] Updating Azure Container App with new version (%BUILD_TAG%)...
call az containerapp update --name baif-cattleweight-app --resource-group rg-baif-cattleweight --image acrcattleweightai.azurecr.io/baif-cattleweight-app:%BUILD_TAG%

echo ===================================================
echo   Success! Revision %BUILD_TAG% is now live on Azure!
echo ===================================================
pause
