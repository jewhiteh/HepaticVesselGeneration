function tree = readTree(path, plt)
% SCRIPT readTree
%
%  USAGE: readTree(path)
%  Reads a Tree.txt file into matlab and returns tree, where each cell is a
%  list of x,y,z coordinate vectors of each branch.
%
% __________________________________________________________________________
%  VARARGIN
%	path: path to Tree.txt file
%   plt: 1 - plot the tree, 0 - do not plot the tree
%

% ------------------------------- Version 1.0 -------------------------------
%	Author:  Joseph Whitehead
%	Email:     joewhitehead1000@gmail.com
%	Created:  2023-03-23
% __________________________________________________________________________
if nargin < 2
    plt = 0;
end

%read in txt file
counter = 1;
fileID = fopen(path,'r');
tline = fgets(fileID);
tree = cell(400000,1);
while tline ~= -1
    temp = str2num(tline);
    temp = reshape(temp,[3 length(temp)/3]);
    tree{counter} = temp + 1;
    counter = counter + 1;
    tline = fgets(fileID);
end
fclose(fileID);
tree = tree(1:counter-1);

%plot
if plt
    %get Radii
    counter = 1;
    path = char(path);
    temp = strsplit(path,filesep);
    if length(temp{end}) == 9
        num = temp{end}(5);
    else
        num = temp{end}(5:6);
    end
    fileID = fopen([path(1:end-9) 'Radii' num '.txt' ],'r');
    tline = fgets(fileID);
    radii = cell(400000,1);
    while tline ~= -1
        radii{counter} = str2num(tline);
        counter = counter + 1;
        tline = fgets(fileID);
    end
    fclose(fileID);
    radii = radii(1:counter-1);


    %plot
    f = figure;
    f.Name = 'Vessel Centerline Tree';
    hold on;
    for i = 1:length(tree)
        plot3(tree{i}(1,:),tree{i}(2,:),tree{i}(3,:),'LineWidth',3 * radii{i} / radii{1},'Color','r')
    end
    hold off;
    drawnow();
    axis equal;
end

end