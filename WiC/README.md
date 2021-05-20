# 3 семинар по NLP
## Задание:
### 1. запустить бейслайн (1 балл)
Файл: WiC.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q8ZmQuuvOBiy5qvlxdDdKhymdFM7uAD7?usp=sharing)
### 2. эксплоративный анализ данных (статистика по корпусу: сколько есть различных слов, как распределены части речи и т.д.) (1 балл)
Файл: WiC_eda.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zdpfSPPz9OhO2xNmx2nsH-1zWCr3wwFs?usp=sharing)
### 3. подберите более подходящие предобученные эмеддинги (2 балла)
Использовал RoBERTa, в WiC.ipynd, идет после байслайна.
Файл: WiC.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q8ZmQuuvOBiy5qvlxdDdKhymdFM7uAD7?usp=sharing)
### 4. попробуйте дополнить эмбеддинги для таргет-слов cls-токенами (1 балл)
Не сделал
### 5. используйте косинусную близость вместо линейных слоев для определения, имеют ли таргет-слова одно значение или разные (2 балла)
Файл: WiC_CosSim.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FuE4g3qjMSDDMx_RJNIZnP07hD9tVNIm?usp=sharing)

### 6. предложите свою архитектуру, дающую улучшение (3 балла)
Использовал подход из NLI. Брал 2 эмбеддинга и проходился по каждому biGRU, потом смотрел CosSim. Качество выросло
Файл: WiC_gru.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PO3jEkVRZH3IIlMev2tL0hQt_whu4q6O?usp=sharing)

### 7. используйте в качестве обучающих данных датасет RUSSE, в качестве модели Rubert (3 балла )
Файл: RUSSE.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1POO6TbUVc1p7Y11fJTDDT5A5kvHbRDV-?usp=sharing)
В комете стоит выбрать RUSSE View
### 8. обучите многоязычную модель на en-en данных, и проверьте, как она работает на других парах, в том числе кросс-язычных (en-ru, en-fr и т.д.) (2 балла)
Самое интересное задание, я решил найти слои, с помощью которых лучше всего происходит мультиязычный перенос. Я думал, для каждой языковой группы будут свои слоя, но нет. 6-9 слой дает качество даже больше, чем все слои и 4 последних.
Файл: WiC_Multilingual.ipynd [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ri_cv7p7NvYGIdyxtCyWqT3Z5QyYeQHM?usp=sharing)
В комете стоит выбрать Multilingual View
### 9. визуализация процесса обучения с помощью TensorBoard (1 балл)
Использовал комету для логов: [ссылка](https://www.comet.ml/danildmitriev1999/3-sem/view/yHegq95QGq3sN3wz1bziDtosl)