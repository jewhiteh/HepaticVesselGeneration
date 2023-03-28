function ph = GenerateLiverVolumn(seg, InputRes)
% SCRIPT GenerateLiverVolumn 
%
%  USAGE: GenerateLiverVolumn(liverVolumn, InputRes)
%  Registers liveVolumn to a Phantom.dat in the BloodDemandMap Folder.
%  Saves a BloodDemandMap phantom. 
%
% __________________________________________________________________________
%  VARARGIN
%	liverVolumn: 3D binary liver volumn (1 - liver, 0 - background)
%   InputRes: 3x1 vector of the resolution in the [x,y,z] dimension (in
%   [mm])
%

% ------------------------------- Version 1.0 -------------------------------
%	Author:  Joseph Whitehead
%	Email:     joewhitehead1000@gmail.com
%	Created:  2023-03-23
% __________________________________________________________________________
if ~isa(seg,'logical')
    error('Please input a logical for the segmentation,');
end
if size(seg,3) == 1
    error('Expected a 3D input for the segmentation.');
end
if length(InputRes) ~= 3
    error('Please input resolution as a 3-element array.');
end


%Get folder to save Phantom
path = cd;
path = fullfile(path(1:strfind(cd,'Vessel_sim_GPU')+13),'BloodDemandMap');

%Initiliaze
f = waitbar(0, 'Starting');
seg = seg > 0;

%Open Blood demand map
fileID = fopen(fullfile(path,'Phantom.dat'),'r');
ph = fread(fileID, strcat('uint8','=>','uint8'), 'l');
ph = reshape(ph, [512 512 512 10]);
fclose(fileID);
map = uint8(ph(:,:,:,10) > 0);

%Get resolutions (mm)
mapSize = [0.77, 0.77, 0.77];

%scale to the same resoltion as the blood demand map
sz = round(size(seg) .*  InputRes ./ mapSize);
seg = imresize3(seg, sz,'nearest');


centroidMap = regionprops(map>0,'Centroid');
centroid_seg = regionprops(seg>0,'Centroid');

%register
Rfixed = imref3d(size(map),mapSize(2),mapSize(1),mapSize(3));
Rmoving = imref3d(size(map),mapSize(2),mapSize(1),mapSize(3));

%Initial rigid translation to register centroids
waitbar(0, f, sprintf('Intial rigid registration...'));
a_mat = diag([1 1 1 1]);
a_mat(4,1:3) = centroid_seg(1).Centroid-centroidMap(1).Centroid;
a_mat(4,1:3) = a_mat(4,1:3).*mapSize;
tform_trans = affine3d(a_mat);
map = imwarp(map,Rmoving, tform_trans,'OutputView',Rmoving);

%Affine registration
waitbar(0, f, sprintf('Affine registration (this may take a few minutes)...'));
[optimizer, metric] = imregconfig('monomodal');
optimizer.MaximumIterations = 10;
tform = imregtform(uint8(map>0),Rmoving,uint8(seg>0),Rfixed,'affine',optimizer,metric,'DisplayOptimization',true);
movingRegisteredVolume = imwarp(map,Rmoving, tform,'OutputView',Rfixed);
disp(['Dice after affine registration:' num2str(dice(seg>0,movingRegisteredVolume>0))]);


%Deformable registration
waitbar(0, f, sprintf('Deformable registration (this may take a few minutes)...'));
[D,movingRegisteredVolume] = imregdemons(uint8(movingRegisteredVolume>0),uint8(seg>0));
disp(['Dice after affine and deformable registration:' num2str(dice(seg>0,movingRegisteredVolume>0))]);

%Perform transform on each phantom and save it out
for i = 1:10
    waitbar(i/10, f, sprintf('Progress: %d %%', floor(i/10*100)));
    temp = imwarp(single(ph(:,:,:,i)),Rmoving, tform_trans,'OutputView',Rfixed);
    temp = imwarp(temp,Rmoving, tform,'OutputView',Rfixed);
    ph(:,:,:,i) = uint8(imwarp(temp, D));
end
waitbar(i/10, f, sprintf('Saving...'));
%Save Phantom
savePhantom(fullfile(path,'PhantomCustom.dat'),uint8(ph),'uint8');
disp(['Succesfully created a liver blood demand map and saved it as ' num2str(fullfile(path,'PhantomCustom.dat'))]);
disp('Saved at a resolution of [0.77, 0.77, 0.77] mm');
end