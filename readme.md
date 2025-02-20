
*Задача:*

Разработать скрипт, который будет с адекватным качеством определять общую эмоциональную окраску текста (“позитивная”-”негативная”)(качество может быть ограничено лимитированным временем). 
Обучающий датасет не предоставляется. 

*Использование:*

[Google Colab Notebook](https://colab.research.google.com/drive/1bg_J9fY9CvGYr_e5MKPzlCFaXMiGjAcQ?usp=sharing)

*Данные:*

Было решено ради интереса использовать русский язык, хотя с русскими корпусами все очень плохо. Далее использовались:

1. http://study.mokoron.com/ - корпус сообщений из твитерра

2. http://text-machine.cs.uml.edu/projects/rusentiment/ - корпус сообщений из vk.
Cкачивание доступно здесь: https://gitlab.com/kensand/rusentiment/tree/master/Dataset

3. https://www.kaggle.com/c/sentiment-analysis-in-russian/overview - корпус новостей

*Структура проекта:*

```./train.sh - запуск обучения ```. Исполняется файл подготовки данных и запускаются скрипты обучения. В качестве параметров передаются пути к данным / желаемые пути к чекпоинтам моделей.

*Модели:*
- Для baseline используется препроцессинг + tf_idf + logreg - служит для того, чтобы познакомиться с данными ( получившаяся точность ~ -0.72 )

- bert. Добучается модель Conversational RuBERT, Russian от DeepPavlov http://docs.deeppavlov.ai/en/master/features/models/bert.html ( получившаяся точность ~ 0.87 )

Веса доступны через Releases.

*Возможные улучшения, если бы это был реальный проект:*

1. Определить точнее область и цель. Дообучать на данных из этой области. ( Т.к. на английском доступно больше, то можно попробовать переводить корпуса ).

2. Улучшить препроцессинг. Попробовать другие модели ( LSTM, CNN, word2vec ). Использовать аугментации.

3. Добавить класс neutral / другое.

4. Учить дольше

5. Ансамбли.

