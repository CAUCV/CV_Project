## 실행 방법

#### 프로그램 설치

- Git v2.28
- Python v3.9
- Visual Studio Code

#### 1. 프로젝트 다운로드

```shell
> git clone https://github.com/CAUCV/CV_Project.git
> cd CV_Project
> code .
```

프로젝트를 다운로드 받고 해당 폴더로 이동해서 VSCode를 실행한다.

VSCode를 실행하면 오른쪽 아래에 '권장 확장 프로그램 설치' 알림이 뜬다.

#### 2. 브랜치 이동 및 의존 패키지 설치

```shell
> git checkout (브랜치 이름)
> python3 -m pip install --upgrade pip
> python3 -m pip install -r Project_1/requirements.txt
```

이동하고 싶은 브랜치로 이동한 후 프로젝트에 필요한 외부 패키지를 설치한다.

#### 3. OpenCV 실행

```shell
> python3 Project_1/project1.py
```
