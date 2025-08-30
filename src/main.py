import uvicorn
from routes import app

if __name__ == "__main__":
    # uvicorn으로 FastAPI 실행
    uvicorn.run(
        "route:app",        # route.py 안의 app 객체 실행
        host="0.0.0.0",     # 외부 접속 허용
        port=8000,          # 포트 번호
        reload=True         # 코드 변경 시 자동 reload
    )