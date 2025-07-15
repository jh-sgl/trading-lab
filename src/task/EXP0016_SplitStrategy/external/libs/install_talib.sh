#!/bin/bash

# TA-Lib 설치 스크립트
# 사용법: ./install_talib.sh [설치 경로]
# 예: ./install_talib.sh /home/user

# 에러 발생 시 스크립트 중단
set -e

# 현재 스크립트 경로를 기본 설치 경로로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_PATH="$SCRIPT_DIR"

# 인자가 있으면 설치 경로로 사용
if [ $# -eq 1 ]; then
    INSTALL_PATH="$1"
    echo "설치 경로를 $INSTALL_PATH 로 설정합니다."
else
    echo "현재 스크립트 경로 $INSTALL_PATH 를 기본 설치 경로로 사용합니다."
    echo "다른 경로를 원하시면: $0 [설치 경로] 형식으로 실행하세요."
fi

# 필요한 디렉토리 생성
mkdir -p "$INSTALL_PATH/ext_libs"

# 현재 작업 디렉토리 저장
CURRENT_DIR=$(pwd)

# TA-Lib 소스 다운로드
echo "TA-Lib 소스 다운로드 중..."
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz

# 압축 해제
echo "압축 해제 중..."
tar -xzf ta-lib-0.6.4-src.tar.gz

# 소스 디렉토리로 이동
echo "TA-Lib 컴파일 중..."
cd ta-lib-0.6.4/

# 컴파일 및 설치
./configure --prefix="$INSTALL_PATH/ext_libs"
make
make install

# 원래 디렉토리로 복귀
cd "$CURRENT_DIR"

# 환경 변수 설정
echo "환경 변수 설정 중..."
export TA_LIBRARY_PATH="$INSTALL_PATH/ext_libs/lib"
export TA_INCLUDE_PATH="$INSTALL_PATH/ext_libs/include"

# Python TA-Lib 설치
echo "Python TA-Lib 패키지 설치 중..."
pip install ta-lib

# 임시 파일 정리
echo "임시 파일 정리 중..."
rm -f ta-lib-0.4.0-src.tar.gz

echo ""
echo "TA-Lib 설치가 완료되었습니다!"
echo "설치 경로: $INSTALL_PATH/ext_libs"
echo ""
echo "현재 세션에서 사용하기 위해 다음 명령어를 실행하세요:"
echo "export TA_LIBRARY_PATH=\"$INSTALL_PATH/ext_libs/lib\""
echo "export TA_INCLUDE_PATH=\"$INSTALL_PATH/ext_libs/include\""