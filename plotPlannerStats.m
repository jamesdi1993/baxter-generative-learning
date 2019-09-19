close all; clear;
mode = 'random';
env = 'one_box_environment';
base_path = "/home/nikhildas/ros_ws/src/baxter_moveit_config/logs/" + env;
folder_path = base_path + '/' + mode;

planners = ["fmt", "vaefmt"];

keys = ["edgeCC", "edgeChecks", "expansionTime", "initialOpenSetupTime", "nodeCounts", ...
    "samplesCollected", "samplingTime", "totalsamplingCC", ...
    "totalTime", "totalCCTime", "totalCC"];
values = zeros(2, length(keys));

% Iterate through each planning problem file in folder;
for i=1:length(planners)
    prefix = folder_path + '/' + 'ompl_' + planners(i);
    files = dir(prefix + '*.log');
    for j = 1:length(files)
        index = 0;
        filename = folder_path + '/' + files(j).name;
        fid = fopen(filename);
        tline = fgetl(fid);
        while ischar(tline)
            index = index + 1;
            substrs = split(tline, ":");
            values(i, index) = values(i, index) + str2double(substrs{2});
            tline = fgetl(fid);
        end
        fclose(fid);
    end
    values(i, :) = values(i, :)/ length(files);
end

% Grep the log file
tmp_files = dir(folder_path + '/' + mode + '*.log');
log_file = tmp_files(1).name;
filename = folder_path + '/' + log_file;
fid = fopen(filename);
tline = fgetl(fid);
start = 0;
while ischar(tline)
    start = start + 1;
    substrs = split(tline, ":");
    values(2 - mod(start, 2), index + idivide(int16(start), int16(2), ...
        'round')) = str2double(substrs{2});
    tline = fgetl(fid);
end
fclose(fid);


% TODO: Add collision-checking time here;
figure('NumberTitle', 'off', 'Name', env);
subplot(2,1,1);
grid on;
time = zeros(4,2);

time(1, :) = values(:, 7); % sampling time
time(2, :) = values(:, 3); % expansionTime
time(3, :) = values(:, 10); % cc time;
time(4, :) = values(:, 9); % total time;

h1 = bar(time); % total time
legend(h1, {'FMT', 'VAE-FMT'});
set(gca, 'xticklabel', {'sampling', 'treeExpansion', 'collisionChecking', 'total'});
grid on
ylabel('seconds')
title("Time");

subplot(2,1,2);
cc = zeros(3,2);
cc(1, :) = values(:, 8); % sample checks
cc(2, :) = values(:, 1); % edge checks
cc(3, :) = values(:, 11); % total checks
h2 = bar(cc); 
grid on
ylabel('counts')
legend(h2, {'FMT', 'VAE-FMT'});
set(gca, 'xticklabel', {'sample', 'edge', 'totalChecks'});
title("Collision Checking");

