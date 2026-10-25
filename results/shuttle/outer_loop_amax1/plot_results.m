%% Shuttle - Outer Loop Control (acceleration level): figures of the reward study (10 seeds)
% Data: data/*.csv and data/study.json written by export_results.py (a_max and source study in study.json).
% All text uses the LaTeX interpreter, i.e. Computer Modern.
% Statistics over seeds: median and interquartile range; individual seeds are shown where relevant.
%
%   plot_results                 % all figures, saved to figures/ (PDF + PNG)
%   plot_results(false)          % draw only
%   plot_results(true, 3)        % seed of the example episode (default 1)
%
% Figures, in the order of the text (Results subsection):
%   fig01 fraction of terminated training episodes
%   fig02 return J and RMSE per reward, one marker per seed (with LQR)
%   fig03a / fig03b / fig03_example_legend  example episode (one goal, seed exampleSeed, 10 s):
%         position, commanded acceleration and their common legend, placed side by side in LaTeX
%   fig05 mean over the 200 goals vs time: (a) position error, (b) commanded acceleration (with LQR)
%   fig06 value loss during training (optional)
%   fig07 final error vs equilibrium offset of the policy
% Requires MATLAB R2020b or later.

function plot_results(saveFigures, exampleSeed)
if nargin < 1, saveFigures = true; end
if nargin < 2, exampleSeed = 1; end
here = fileparts(mfilename('fullpath'));
dataDir = fullfile(here, 'data');
figDir  = fullfile(here, 'figures');
if saveFigures && ~exist(figDir, 'dir'), mkdir(figDir); end

% Computer Modern: every text object through the LaTeX interpreter
names = {'defaultTextInterpreter', 'defaultAxesTickLabelInterpreter', 'defaultLegendInterpreter', ...
         'defaultColorbarTickLabelInterpreter', 'defaultAxesFontName', 'defaultTextFontName', ...
         'defaultLegendFontName', 'defaultAxesFontSize', 'defaultTextFontSize', 'defaultLegendFontSize'};
previous = cellfun(@(n) get(groot, n), names, 'UniformOutput', false);
for k = 1:4, set(groot, names{k}, 'latex'); end      % the first four are interpreters
set(groot, 'defaultAxesFontName', 'Computer Modern', 'defaultTextFontName', 'Computer Modern', ...
           'defaultLegendFontName', 'Computer Modern', 'defaultAxesFontSize', 9, 'defaultTextFontSize', 9, ...
           'defaultLegendFontSize', 9);
restore = onCleanup(@() restoreDefaults(names, previous));

info = jsondecode(fileread(fullfile(dataDir, 'study.json')));
aMax = info.a_max;
M  = readtable(fullfile(dataDir, 'run_metrics.csv'), 'TextType', 'string');
T  = readtable(fullfile(dataDir, 'training_curves.csv'), 'TextType', 'string');
C  = readtable(fullfile(dataDir, 'eval_curves.csv'), 'TextType', 'string');
EP = readtable(fullfile(dataDir, 'example_episode.csv'), 'TextType', 'string');

S = style();
rewards = S.rewards;                                  % trained policies, in the order of the tables
lqrJ = M.J_quad(M.experiment == "lqr");
lqrRmse = M.rmse_m(M.experiment == "lqr");

%% fig01 - fraction of training episodes that end by termination
f = newFigure(S, 17, 7.5);
ax = axes(f); hold(ax, 'on');
h = gobjects(0);
for i = 1:numel(rewards)
    [x, Y] = bySeed(T(T.experiment == rewards(i), :), 'iteration', 'died', (1:200)', 3);
    h(i) = medianBand(ax, x, Y, S.color(i, :), S.line{i});
end
xlim(ax, [0 200]); ylim(ax, [0 1.02]);
xlabel(ax, 'PPO iteration'); ylabel(ax, 'Terminated episodes (fraction)');
formatAxis(ax, S);
axisLegend(ax, h, S.labels);
saveFigure(f, figDir, 'fig01_training_terminations', saveFigures);

%% fig02 - return of r^Q and position RMSE per reward (one marker per seed), LQR as reference
f = newFigure(S, 15, 7.2);
tl = tiledlayout(f, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
fields = {'J_quad', 'rmse_m'};
ylabs  = {'Return $J$ of $r^{\mathrm{Q}}$', 'Position RMSE [m]'};
ttls   = {'(a) Return', '(b) RMSE'};
refs   = [lqrJ, lqrRmse];
for k = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    for i = 1:numel(rewards)
        v = M.(fields{k})(M.experiment == rewards(i));
        seedStrip(ax, i, v, S.color(i, :), 'o');
    end
    hl = yline(ax, refs(k), '--', 'Color', 'k', 'LineWidth', 1.1);
    formatAxis(ax, S);
    set(ax, 'XTick', 1:numel(rewards), 'XTickLabel', S.shortLabels, 'XTickLabelRotation', 0);
    xlim(ax, [0.5 numel(rewards) + 0.5]); ax.XGrid = 'off';
    ylabel(ax, ylabs{k}); title(ax, ttls{k}, 'FontWeight', 'normal');
    if k == 1   % LQR legend inside the return axes, lower left
        legend(ax, hl, {'LQR'}, 'Location', 'southwest', 'Box', 'off', 'Interpreter', 'latex');
    end
end
saveFigure(f, figDir, 'fig02_return_rmse', saveFigures);

%% fig03 - example episode: same goal, one seed per reward, and the LQR (position | acceleration)
shown = [rewards, "lqr"];
E = cell(1, numel(shown));
for i = 1:numel(shown)
    seed = exampleSeed * (shown(i) ~= "lqr");
    E{i} = EP(EP.experiment == shown(i) & EP.seed == seed, :);
end
exampleFigure(S, E, shown, aMax, figDir, exampleSeed, saveFigures);

%% fig05 - mean over the 200 goals versus time (episodes still running): position error and commanded acceleration
f = newFigure(S, 17, 10);
tl = tiledlayout(f, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
fields = {'err_mean', 'u_mean'};
ylabs  = {'Mean $\|\mathbf{e}_p\|$ [m]', 'Mean $\|\mathbf{u}\|$ [m/s$^2$]'};
ttls   = {'(a) Position error', '(b) Commanded acceleration'};
grid10 = (0.05:0.1:9.95)';
L = C(C.experiment == "lqr", :);
for k = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    h = gobjects(0);
    for i = 1:numel(rewards)
        [x, Y] = bySeed(C(C.experiment == rewards(i), :), 't', fields{k}, grid10, 1);
        h(i) = medianBand(ax, x, Y, S.color(i, :), S.line{i});
    end
    h(end + 1) = plot(ax, L.t, L.(fields{k}), '--', 'Color', 'k', 'LineWidth', 1.3); %#ok<AGROW>
    xlim(ax, [0 10]); ylabel(ax, ylabs{k}); title(ax, ttls{k}, 'FontWeight', 'normal');
    formatAxis(ax, S);
    if k == 1
        set(ax, 'YScale', 'log'); ylim(ax, [0.005 2.5]); ax.XTickLabel = [];
    else
        ylim(ax, [0 inf]); xlabel(ax, 'Time [s]');
    end
end
lgd = legend(h, [S.labels, {'LQR'}], 'Orientation', 'vertical', 'Box', 'off', 'Interpreter', 'latex');
lgd.Layout.Tile = 'east';
saveFigure(f, figDir, 'fig05_mean_error_acceleration', saveFigures);

%% fig06 - value loss during training (log scale)
f = newFigure(S, 17, 7.5);
ax = axes(f); hold(ax, 'on');
h = gobjects(0);
for i = 1:numel(rewards)
    [x, Y] = bySeed(T(T.experiment == rewards(i), :), 'iteration', 'value_loss', (1:200)', 5);
    h(i) = medianBand(ax, x, Y, S.color(i, :), S.line{i});
end
set(ax, 'YScale', 'log'); xlim(ax, [0 200]);
xlabel(ax, 'PPO iteration'); ylabel(ax, 'Value loss');
formatAxis(ax, S);
axisLegend(ax, h, S.labels);
saveFigure(f, figDir, 'fig06_value_loss', saveFigures);

%% fig07 - final error of the evaluation versus equilibrium offset of the policy (surviving policies)
f = newFigure(S, 15, 9);
ax = axes(f); hold(ax, 'on');
h = gobjects(0);
top = 0;
for i = 1:numel(rewards)
    R = M(M.experiment == rewards(i) & M.survival > 0.5, :);
    if isempty(R), continue; end
    h(end + 1) = scatter(ax, R.equilibrium_offset_m, R.final_err_m, 22, S.color(i, :), S.marker{i}, ...
        'filled', 'MarkerFaceAlpha', 0.8); %#ok<AGROW>
    top = max([top; R.equilibrium_offset_m; R.final_err_m]);
end
plot(ax, [0 1.05 * top], [0 1.05 * top], ':', 'Color', [0.3 0.3 0.3], 'HandleVisibility', 'off');
axis(ax, 'equal'); xlim(ax, [0 1.05 * top]); ylim(ax, [0 1.05 * top]);
xlabel(ax, 'Equilibrium offset $\|\mathbf{e}_p^{*}\|$ [m]'); ylabel(ax, 'Final error (evaluation) [m]');
formatAxis(ax, S);
labels = S.labels(arrayfun(@(r) any(M.experiment == r & M.survival > 0.5), rewards));
legend(ax, h, labels, 'Location', 'eastoutside', 'Orientation', 'vertical', 'Box', 'off', 'Interpreter', 'latex');
saveFigure(f, figDir, 'fig07_equilibrium_offset', saveFigures);

%% console summary (median over seeds)
disp(groupsummary(M, 'experiment', 'median', {'survival', 'final_err_m', 'rmse_m', 't20_s', 'J_quad', ...
    'equilibrium_offset_m', 'value_loss_peak'}));
end

%% =============================================================== helpers
function restoreDefaults(names, values)
for k = 1:numel(names), set(groot, names{k}, values{k}); end
end

function q = quantiles(Y, p)
% Quantiles of each row of Y (linear interpolation; no Statistics Toolbox needed).
Y = sort(Y, 2);
n = size(Y, 2);
pos = 1 + p * (n - 1);
lo = floor(pos); hi = ceil(pos); w = pos - lo;
q = Y(:, lo) .* (1 - w) + Y(:, hi) .* w;
end

function S = style()
S.font = 9;
S.rewards = ["quad", "alive", "death", "exp", "alive_death", "alive_death200"];
S.labels = {'$r^{\mathrm{Q}}$', '$r^{\mathrm{A}}$', '$r^{\mathrm{D}}$', '$r^{\mathrm{E}}$', ...
            '$r^{\mathrm{AD}}$, $d = -500$', '$r^{\mathrm{AD}}$, $d = -200$'};
S.shortLabels = {'$r^{\mathrm{Q}}$', '$r^{\mathrm{A}}$', '$r^{\mathrm{D}}$', '$r^{\mathrm{E}}$', ...
                 '$r^{\mathrm{AD}}_{500}$', '$r^{\mathrm{AD}}_{200}$'};
S.color = [0.55 0.55 0.55;     % Q
           0.10 0.30 0.65;     % A
           0.85 0.33 0.10;     % D
           0.25 0.55 0.30;     % E
           0.55 0.25 0.60;     % AD, -500
           0.80 0.15 0.35];    % AD, -200
S.line = {'-', '-', '-', '-', '--', '-'};
S.marker = {'o', 's', 'd', '^', 'v', 'o'};
end

function f = newFigure(S, widthCm, heightCm)
f = figure('Units', 'centimeters', 'Color', 'w', 'WindowStyle', 'normal');
f.Position = [2 2 widthCm heightCm];
set(f, 'DefaultAxesFontSize', S.font);
end

function formatAxis(ax, S)
set(ax, 'FontName', 'Computer Modern', 'FontSize', S.font, 'TickLabelInterpreter', 'latex', 'Box', 'on', 'TickDir', 'in', 'LineWidth', 0.6);
grid(ax, 'on'); ax.GridAlpha = 0.15;
end

function axisLegend(ax, h, names)
legend(ax, h, names, 'Location', 'eastoutside', 'Orientation', 'vertical', 'Box', 'off', 'Interpreter', 'latex');
end

function [x, Y] = bySeed(Tab, xField, yField, x, smoothWin)
% One column per seed, resampled on x (seeds may log different numbers of points).
seeds = unique(Tab.seed);
Y = nan(numel(x), numel(seeds));
for s = 1:numel(seeds)
    r = Tab(Tab.seed == seeds(s), :);
    [xs, first] = unique(r.(xField));
    ys = r.(yField)(first);
    ok = ~isnan(ys);
    if nnz(ok) < 2, continue; end
    if smoothWin > 1, ys(ok) = smoothdata(ys(ok), 'movmean', smoothWin); end
    Y(:, s) = interp1(xs(ok), ys(ok), x, 'linear', NaN);
end
end

function h = medianBand(ax, x, Y, color, lineStyle)
% Median over seeds (line) and interquartile range (shaded); times without all seeds are dropped.
valid = all(~isnan(Y), 2);
x = x(valid); Y = Y(valid, :);
q = quantiles(Y, [0.25 0.5 0.75]);
fill(ax, [x; flipud(x)], [q(:, 1); flipud(q(:, 3))], color, 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
h = plot(ax, x, q(:, 2), lineStyle, 'Color', color, 'LineWidth', 1.4);
end

function seedStrip(ax, x0, v, color, marker)
% Individual seeds (open markers) with the median (filled) and interquartile range (bar).
x = x0 + linspace(-0.12, 0.12, numel(v))';
scatter(ax, x, v, 14, color, marker, 'HandleVisibility', 'off');
q = quantiles(v(:)', [0.25 0.5 0.75]);
plot(ax, [x0 x0], q([1 3]), '-', 'Color', color, 'LineWidth', 1.6, 'HandleVisibility', 'off');
plot(ax, x0, q(2), marker, 'Color', color, 'MarkerFaceColor', color, 'MarkerSize', 7, 'HandleVisibility', 'off');
end

function exampleFigure(S, E, shown, aMax, figDir, seed, doSave)
% Example episode as three files, arranged side by side in LaTeX (subfigures) with the legend above:
%   fig03a_example_position_seed<seed>      position per axis, with the reference (black dotted)
%   fig03b_example_acceleration_seed<seed>  commanded acceleration per axis, with +-a_max (grey dotted)
%   fig03_example_legend                    the common legend, horizontal
% Rewards solid, LQR black dashed. Each panel has the size of the thrust-and-moment episode figures (15 x 10 cm).
tEnd = 10;                                  % [s] time window shown (the whole episode)
panelW = 15; panelH = 10;                   % [cm] each panel, as the episode figures of the thrust-and-moment study
cols = {{'x', 'y', 'z'}, {'ux', 'uy', 'uz'}};
ylabs = {{'$p_x$ [m]', '$p_y$ [m]', '$p_z$ [m]'}, {'$u_x$ [m/s$^2$]', '$u_y$ [m/s$^2$]', '$u_z$ [m/s$^2$]'}};
refs = {'goal_x', 'goal_y', 'goal_z'};
files = {sprintf('fig03a_example_position_seed%d', seed), sprintf('fig03b_example_acceleration_seed%d', seed)};
for c = 1:2
    f = newFigure(S, panelW, panelH);
    tl = tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:3
        ax = nexttile(tl); hold(ax, 'on');
        field = cols{c}{k};
        for i = 1:numel(shown)
            Ei = E{i};
            if isempty(Ei), continue; end
            if shown(i) == "lqr"
                plot(ax, Ei.t, Ei.(field), '--', 'Color', 'k', 'LineWidth', 1.3);
            else
                plot(ax, Ei.t, Ei.(field), S.line{i}, 'Color', S.color(i, :), 'LineWidth', 1.2);
            end
        end
        if c == 1
            plot(ax, E{end}.t, E{end}.(refs{k}), ':', 'Color', 'k', 'LineWidth', 1.1);
        else
            yline(ax, aMax, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.1);
            yline(ax, -aMax, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.1);
            ylim(ax, 1.15 * [-aMax aMax]);
        end
        xlim(ax, [0 tEnd]); ylabel(ax, ylabs{c}{k}); formatAxis(ax, S);
        if k < 3
            ax.XTickLabel = [];
        else
            xlabel(ax, 'Time [s]');
        end
    end
    saveFigure(f, figDir, files{c}, doSave);
end

% legend only: invisible axes with one line per entry, in the same styles
f = newFigure(S, 2 * panelW, 1.0);
ax = axes(f, 'Visible', 'off'); hold(ax, 'on');
h = gobjects(0);
for i = 1:numel(shown)
    if shown(i) == "lqr"
        h(end + 1) = plot(ax, NaN, NaN, '--', 'Color', 'k', 'LineWidth', 1.3); %#ok<AGROW>
    else
        h(end + 1) = plot(ax, NaN, NaN, S.line{i}, 'Color', S.color(i, :), 'LineWidth', 1.2); %#ok<AGROW>
    end
end
h(end + 1) = plot(ax, NaN, NaN, ':', 'Color', 'k', 'LineWidth', 1.1);
h(end + 1) = plot(ax, NaN, NaN, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.1);
names = [S.labels(1:numel(shown) - 1), {'LQR', 'Reference', '$\pm a_{\max}$'}];
lgd = legend(ax, h, names, 'Orientation', 'horizontal', 'NumColumns', numel(names), 'Box', 'off', ...
             'Interpreter', 'latex', 'FontSize', S.font);
lgd.Units = 'normalized'; lgd.Position(1:2) = [0.5 - lgd.Position(3) / 2, 0.5 - lgd.Position(4) / 2];
saveFigure(f, figDir, 'fig03_example_legend', doSave);
end

function saveFigure(f, figDir, name, doSave)
% PNG with exportgraphics; PDF with print -painters (same path as the manual export of the figure window):
% the LaTeX text becomes vector outlines instead of selectable text set in a substitute font.
if ~doSave, return; end
drawnow;
if ~isvalid(f), warning('Figure %s was closed before it could be saved.', name); return; end
exportgraphics(f, fullfile(figDir, [name '.png']), 'Resolution', 300);
set(f, 'PaperUnits', 'centimeters');
pos = get(f, 'Position');
set(f, 'PaperSize', pos(3:4), 'PaperPosition', [0 0 pos(3:4)]);
print(f, fullfile(figDir, [name '.pdf']), '-dpdf', '-painters');
end
