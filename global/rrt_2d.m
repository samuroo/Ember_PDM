clear all;  % Clear workspace
close all;  % Close all figures
clc;        % Clear command window

%% Define environment

bounds = [0, 100;   % x-axis limits
          0, 100]; % y-axis limits

% Define obstacles as [x_center, y_center, radius]
obstacles = [20 20 8;
             46 32 12;
             72 60 9];

% Define start and end points
startp = [5, 5];
endp   = [95, 95];

% Create figure
figure; hold on; axis equal;
xlim(bounds(1,:)); ylim(bounds(2,:));
title('RRT Planner'); xlabel('X'); ylabel('Y');

% Plot obstacles as circles
for i = 1:size(obstacles,1)
    viscircles(obstacles(i,1:2), obstacles(i,3), 'Color','k');
end

% Plot start and goal points
plot(startp(1), startp(2), 'go', 'MarkerSize',10,'LineWidth',2); % Start in green
plot(endp(1), endp(2), 'ro', 'MarkerSize',10,'LineWidth',2);     % Goal in red

drawnow; % Update the figure

%% Separate definitions

it_max = 10000; % Maximum number of iterations
rob_r = 5;      % Robot radius for collision checking
increment = 3;  % Step size (distance to move toward random sample)

V = [startp];   % List of vertices (nodes)
E = [];         % List of edges (connections between vertices)

%% Main RRT loop

for i = 1:it_max

    % Sample a random point within the bounds
    randp = [rand() * bounds(1, 2), rand() * bounds(2, 2)];

    % Compute distances from all vertices to random point
    distances = vecnorm(V - randp, 2, 2);

    % Find nearest vertex
    [~, nearest_idx] = min(abs(distances));

    % Compute direction vector (normalized)
    normal = (randp - V(nearest_idx,:)) / distances(nearest_idx);

    % Compute new point at fixed increment toward random point
    new_p = V(nearest_idx,:) + increment * normal;

    %% Collision check

    n_points = 10; % Number of points along line segment for collision checking
    dx = linspace(0, increment, n_points); % Parameterize the line

    % Points along the line from nearest vertex to new point
    x_line = V(nearest_idx,1) + dx .* normal(1, 1);
    y_line = V(nearest_idx,2) + dx .* normal(1, 2);
    E_line = [x_line(:), y_line(:)];

    [rows_obs, ~] = size(obstacles);
    E_dists = zeros(rows_obs*n_points, 1);

    % Compute distance from each point along line to all obstacles
    for j = 1:n_points
        E_dists((j-1)*rows_obs+1 : j*rows_obs) = vecnorm(E_line(j,:) - obstacles(:,1:2), 2, 2);
    end

    % Required clearance from obstacles (radius + robot radius)
    clearance = obstacles(:,3) + rob_r;

    % Check if new point and path segment are collision-free
    if all(vecnorm(new_p - obstacles(:,1:2),2,2) > clearance) && ...
       all(E_dists > kron(clearance, ones(n_points,1)))

        % Add new point to vertices
        V = [V; new_p];

        % Add edge from nearest vertex to new point
        E = [E; nearest_idx, size(V, 1)];

        % Plot the new edge and point
        plot([V(nearest_idx,1), new_p(1)], [V(nearest_idx,2), new_p(2)], 'b-');
        plot(new_p(1, 1), new_p(1, 2), 'b.', 'MarkerSize', 5);
        drawnow;

        % Check if goal is reached (within increment distance)
        if norm(new_p - endp) < increment
            % Connect final point to goal
            V = [V; endp];
            E = [E; size(V,1)-1, size(V,1)];
            goal_idx = size(V,1);
            disp("Goal reached!");
            break;
        end
    end
end

%% Backtrack path

path = goal_idx;
current = goal_idx;

% Trace back from goal to start using edges
while current ~= 1
    parent = E(E(:,2) == current, 1); % Find parent vertex
    current = parent;
    path = [current, path]; % Build path from start to goal
end

%% Highlight final path

for k = 1:length(path)-1
    plot([V(path(k),1), V(path(k+1),1)], ...
         [V(path(k),2), V(path(k+1),2)], 'r-', 'LineWidth', 3); % Red thick line
end
