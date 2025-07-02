% 定义四个法向量
a = [0.2251 1.6 -0.9743];   % 第一个法向量
b = [-0.9743 1.6 -0.2251];   % 第二个法向量
c = [-0.2251 1.6 0.9743];   % 第三个法向量
d = [0.9743 1.6 0.2251];  % 第四个法向量
e = [0.2251 1.6 -0.9743];

% 添加虚拟点
epsilon = 1e-6;
% 绘制y轴线条
plot3([0 0], [epsilon -epsilon], [0 0], 'k--', 'linewidth', 1.5);
hold on;

% 将法向量单位化
a = a/norm(a);
b = b/norm(b);
c = c/norm(c);
d = d/norm(d);
e = e/norm(e);

% 绘制箭头
quiver3(0, 0, 0, a(1), a(2), a(3), 'r', 'linewidth', 1.5);
quiver3(0, 0, 0, b(1), b(2), b(3), 'g', 'linewidth', 1.5);
quiver3(0, 0, 0, c(1), c(2), c(3), 'b', 'linewidth', 1.5);
quiver3(0, 0, 0, d(1), d(2), d(3), 'm', 'linewidth', 1.5);
quiver3(0, 0, 0, e(1), e(2), e(3), 's', 'linewidth', 1.5);

axis equal;
grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
