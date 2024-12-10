import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os


# Определяет, находится ли точка p внутри полуплоскости, образованной ребром (cp1, cp2)
def inside(p, cp1, cp2):
    return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])


# Вычисляет точку пересечения двух отрезков (cp1, cp2) и (s, e)
def intersection(cp1, cp2, s, e):
    dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
    dp = (s[0] - e[0], s[1] - e[1])
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    return (n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3


# Алгоритм Sutherland-Hodgman для отсечения многоугольника
def sutherland_hodgman(subject_pol, clip_pol):
    output_list = subject_pol
    cp1 = clip_pol[-1]
    for cp2 in clip_pol:
        input_list = output_list
        output_list = []
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output_list.append(intersection(cp1, cp2, s, e))
                output_list.append(e)
            elif inside(s, cp1, cp2):
                output_list.append(intersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
    return output_list


def lianga_barsky(segments, clipper):
    clipped_segments = []
    
    if not clipper:
        return clipped_segments

    # Извлекаем координаты всех точек отсекателя
    clip_points = clipper[0]
    xs = [p[0] for p in clip_points]
    ys = [p[1] for p in clip_points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    for segment in segments:
        p1, p2 = segment
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1

        p = [-dx, dx, -dy, dy]
        q = [x1 - x_min, x_max - x1, y1 - y_min, y_max - y1]

        u1 = 0.0
        u2 = 1.0
        accept = True

        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    accept = False
                    break  # Отрезок параллелен границе и вне отсекателя
                else:
                    continue  # Отрезок параллелен границе и внутри отсекателя
            t = qi / pi
            if pi < 0:
                if t > u2:
                    accept = False
                    break
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    accept = False
                    break
                if t < u2:
                    u2 = t

        if accept and u1 <= u2:
            clipped_p1 = (x1 + u1 * dx, y1 + u1 * dy)
            clipped_p2 = (x1 + u2 * dx, y1 + u2 * dy)
            clipped_segments.append((clipped_p1, clipped_p2))

    return clipped_segments




# Функция для отрисовки многоугольников
def plot_polygons(subject_pol, clip_pol, clipped_pol):
    plt.figure()
    plt.fill(*zip(*subject_pol), edgecolor='r', fill=False, linewidth=2, label='Subject Polygon')
    plt.scatter(*zip(*subject_pol), color='r', zorder=5)
    plt.fill(*zip(*clip_pol), edgecolor='b', fill=False, linewidth=2, label='Clip Polygon')
    plt.scatter(*zip(*clip_pol), color='b', zorder=5)
    plt.fill(*zip(*clipped_pol), edgecolor='g', fill=False, linewidth=2, label='Clipped Polygon')
    plt.scatter(*zip(*clipped_pol), color='g', zorder=5)
    plt.legend()
    plt.title('Sutherland-Hodgman Polygon Clipping')

def plot_lianga_barski(segments, clipper, clipped_segments):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Рисуем исходные сегменты
    for idx, seg in enumerate(segments):
        x_vals, y_vals = zip(*seg)
        plt.plot(x_vals, y_vals, color='blue', linewidth=2, label='Исходные сегменты' if idx == 0 else "")

    # Рисуем отсекатель
    if clipper:
        # Предполагаем, что clipper - это список, содержащий один кортеж из четырёх точек
        clip_points = clipper[0]
        # Закрываем многоугольник, добавляя первую точку в конец списка
        clipper_closed = list(clip_points) + [clip_points[0]]
        x_clipper, y_clipper = zip(*clipper_closed)
        plt.plot(x_clipper, y_clipper, color='red', linewidth=2, label='Отсекатель')

    # Рисуем отсечённые сегменты
    for idx, seg in enumerate(clipped_segments):
        x_vals, y_vals = zip(*seg)
        plt.plot(x_vals, y_vals, color='green', linewidth=2, label='Отсечённые сегменты' if idx == 0 else "")

    # Настройка легенды, чтобы избежать дублирования меток
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Дополнительные настройки графика
    plt.title('Отсечение Лианга-Барски')
    plt.xlabel('X-ось')
    plt.ylabel('Y-ось')
    plt.grid(True)
    plt.axis('equal')  # Сохраняет одинаковый масштаб по осям
    plt.show()



# Генерация выпуклого многоугольника с использованием ConvexHull
def generate_convex_polygon(n):
    points = np.random.rand(n, 2) * 100
    hull = ConvexHull(points)
    return points[hull.vertices]


# Генерация простого многоугольника в заданной области
def generate_simple_polygon(n, bbox):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = np.random.rand(n) * (max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 0.5)
    points = np.vstack((radius * np.cos(angle), radius * np.sin(angle))).T
    center = np.array([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2])
    points += center
    return points


# Ручной ввод координат
def manual_input(n_subject, n_clip):
    subject_polygon = []
    clip_polygon = []

    print(f"Введите {n_subject} пар координат для произвольного полигона:")
    for i in range(n_subject):
        x = float(input(f"Введите координату x{i + 1} для произвольного полигона: "))
        y = float(input(f"Введите координату y{i + 1} для произвольного полигона: "))
        subject_polygon.append((x, y))

    print(f"Введите {n_clip} пар координат для выпуклого отсекателя:")
    for i in range(n_clip):
        x = float(input(f"Введите координату x{i + 1} для выпуклого отсекателя: "))
        y = float(input(f"Введите координату y{i + 1} для выпуклого отсекателя: "))
        clip_polygon.append((x, y))

    return subject_polygon, clip_polygon

# Чтение входных данных из файла (алгоритм Лианга Барски)
def reading_langaBarski():
    filename = input("введите имя файла с входными данными (если файл в той же папке, просто укажите его имя):\n")
    
    # Получаем путь к файлу, предполагая, что он в той же директории
    file_path = os.path.join(os.getcwd(), filename)

    try:
        with open(file_path, 'r') as f:
            n = int(f.readline())  # Считываем количество отрезков
            segments = []
            for _ in range(n):
                x1, y1, x2, y2 = map(float, f.readline().split())
                segments.append(((x1, y1), (x2, y2)))
            
            # Прямоугольник clipper
            clipper = []
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, f.readline().split())
            clipper.append(((x1, y1), (x2, y2), (x3, y3), (x4, y4)))
        
        return segments, clipper
    
    except FileNotFoundError:
        print(f"Ошибка: файл '{filename}' не найден.")
        return [], []
    except ValueError:
        print("Ошибка: неверный формат данных в файле.")
        return [], []


# Генерация многоугольников
def generate_polygons(n_subject, n_clip):
    clip_polygon = generate_convex_polygon(n_clip)
    bbox = [np.min(clip_polygon[:, 0]) - 50, np.max(clip_polygon[:, 0]) + 50, np.min(clip_polygon[:, 1]) - 50,
            np.max(clip_polygon[:, 1]) + 50]
    subject_polygon = generate_simple_polygon(n_subject, bbox)
    return subject_polygon, clip_polygon


# Основная функция
def main():
    n_clip = 5
    n_subject = 4

    while True:
        choose = input("1.Алгоритм Лианга-Барски\n2.Отсечение выпуклого многоугольника\n")
        if choose == "1":
            segments, clipper = reading_langaBarski()
            break
        elif choose == "2":
            while True:
                choose1 = input("\t1. ручной ввод координат.\n\t2. автоматическая генерация координат.\n")
                if choose1 == "1":
                    subject_polygon, clip_polygon = manual_input(n_subject, n_clip)
                    break
                elif choose == "2":
                    subject_polygon, clip_polygon = generate_polygons(n_subject, n_clip)
                    break
                else:
                    print("неправильный формат ввода\n")
            break
        else:
            print("неправильный формат ввода\n")


    if (choose == "1"):
        print("LiangaBarski Segments:")
        for point in segments:
            print(f"({point[0]}, {point[1]})")

        print("LiangaBarski Clipper")
        # for point in clipper:
        #     print(f"({point[0]}, {point[1]})")
        print(clipper)

        # Применение алгоритма Лианга-Барски
        clipped_segments = lianga_barsky(segments, clipper)

        print(clipped_segments)

        # Отрисовка
        plot_lianga_barski(segments, clipper, clipped_segments)

        

    else:
        print("Clip Polygon Points:")
        for point in clip_polygon:
            print(f"({point[0]}, {point[1]})")

        print("Subject Polygon Points:")
        for point in subject_polygon:
            print(f"({point[0]}, {point[1]})")

        # Применение алгоритма отсечения
        clipped_polygon = sutherland_hodgman(subject_polygon, clip_polygon)

        # Вывод координат вершин многоугольников
        print("Clipped Polygon Points:")
        for point in clipped_polygon:
            print(f"({point[0]}, {point[1]})")

        # Отрисовка всех многоугольников
        plot_polygons(subject_polygon, clip_polygon, clipped_polygon)

    

if __name__ == "__main__":
    main()
