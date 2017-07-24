addpath('/home/zxy/Datasets/Human3.6M/Release-v1.1/H36M');
addpath('/home/zxy/Datasets/Human3.6M/Release-v1.1/H36MBL');
addpath('/home/zxy/Datasets/Human3.6M/Release-v1.1/H36MGUI');
addpath(genpath('/home/zxy/Datasets/Human3.6M/Release-v1.1/utils'));
addpath(genpath('/home/zxy/Datasets/Human3.6M/Release-v1.1/external_utils'));
%addpath(genpath('../'));
node_id = 1
num_nodes = 1
generate_type = 0

raw_data_dir = '/home/zxy/Datasets/Human3.6M/Data/';
istrain = 1;
subject_list = [1,5,6,7,8,9,11];
%subject_list = [11];
action_list = 2:16;
subaction_list = 1:2;
camera_list = 1:4;
%save_data_root = '/opt/visal/data/H36/H36MData/Train_test';
save_data_root = '/home/zxy/Datasets/Human3.6M/Processed/Train/';
bndadjustopt = struct('ratio', 0);
if ~exist(save_data_root,'dir')
    mkdir(save_data_root);
end


num_tasks = length(subject_list) * length( action_list) ...
    * length(subaction_list) * length(camera_list);  
per_node = floor((num_tasks - 1)/num_nodes) + 1;
start_task = (node_id - 1) * per_node + 1;
end_task = node_id * per_node;
tid = 0;

h36m_database = H36MDataBase.instance();

imeta = H36ProjectMeta();
subfolder_format = imeta.subfolder_format;
for s = subject_list
    for act = action_list
        for subact = subaction_list
            for ca = camera_list
                tid = tid + 1;
                if (s == 11 && act == 2 && subact == 2 && ca == 1)
                    continue;
                end
                if (tid < start_task)
                    continue;
                end
                if (tid > end_task)
                    return;
                end
                subfolder = sprintf(subfolder_format, s, act, subact, ...
                                    ca)
                subfolder_full = fullfile(save_data_root, subfolder);
                if exist(fullfile(subfolder_full, 'dir')) == 0
                    mkdir(subfolder_full);
                end
                if generate_type == 3
                    subfolder_mask_full = fullfile(save_fgmask_root, subfolder);
                    if exist(fullfile(subfolder_mask_full, 'dir')) == 0
                        mkdir(subfolder_mask_full);
                    end
                end
                save_meta_name = fullfile(subfolder_full, ...
                                          'matlab_meta.mat');
                if exist(save_meta_name, 'file')
                    fprintf('%s exists, ha ha ha \n', save_meta_name);
                    continue;
                end
                
                vname = h36m_database.getFileName(s, act, subact, ca);
                fprintf('Processing %s in node %d under %s\n', vname, ...
                        node_id, subfolder);
                vobj = H36MVideoDataAccess(fullfile(raw_data_dir, ...
                                                sprintf('S%d', s), ...
                                                'Videos', ...
                                          [vname '.mp4']));
                pname = h36m_database.getFileName(s, act, subact);
                posobj = H36MPoseDataAcess(fullfile(raw_data_dir, ...
                                                    sprintf('S%d', s), ...
                                           'MyPoseFeatures/D3_Positions',...
                                                    [pname ...
                                    '.cdf']));
                sobj = H36MSequence(s, act, subact, ca);
                %cur_occ_rec = occ_record(s, act, subact, ca);
                %masks = get_mask_image(cur_occ_rec, occ_mask_dir, ...
                %                                    imeta.image_size);
                num_frames = sobj.getNumFrames();
                cur_camera = sobj.getCamera(ca);
                cur_subject_model = imeta.get_subject_skelmodel(s);
                mono_joint_list = imeta.mono_joint_list;
                num_joint = length(mono_joint_list);
                % To save space, I keep Y3d_mocap, Y2d_image_body empty
                % 
                %{
                meta_data = struct('X', [], ...             % Empty 
                                   'images_path',[], ...   
                                   'Y3d_mocap', [],...      % Empty    
                                   'Y3d_mono_body',...
                                   zeros(num_joint * 3,num_frames),... 
                                   'Y2d_image_body', [], ... % Empty
                                   'Y2d_bnd_body',...
                                    zeros(num_joint * 2, num_frames),... 
                                   'occ_body', ...
                                   zeros(num_joint, num_frames), ...     
                                   'oribbox', zeros(4, num_frames), ... 
                                   'image_dim', imeta.image_size, ...
                                   'image_adjust_dim', ...
                                   imeta.image_adjust_dim, ...
                                   'mean_image', zeros(imeta.image_adjust_dim),...
                                   'num_images', num_frames, ...
                                   'rgb_meancov', zeros(3,3));  
                %}               
                meta_data = struct('images_path',[], ...   
                                   'Y3d_mono', zeros(num_joint * 3,num_frames),... 
                                   'Y2d', zeros(num_joint * 2, num_frames),...  
                                   'bbox', zeros(4, num_frames), ...
                                   'num_images', num_frames);  
                meta_data.images_path = cell(1, num_frames);
                disp('The block size is')
                disp(size(posobj.Block))
                fprintf('The number of frames is %d\n', ...
                        num_frames);
                % Although not everyone will use it, just load it
                bgmat = load(fullfile(raw_data_dir, ...
                                      sprintf('S%d', s), ...
                                      'MySegmentsMat', 'ground_truth_bs', ...
                                      [vname '.mat']));
                for fr = 1:5:num_frames
                    %fprintf('\rIn Processing %.3f\r',fr/num_frames * 100);
                    %% Estimate the bounding box first 
                    %% Please note: get_boundingbox will use 1x96
                    %% vector as input,never 
                    %[x,y,x1,y1] format
                    pos = posobj.Block(fr,:);
                    fgmask = bgmat.Masks{fr};
                    fgmask_rgb = repmat(fgmask, [1,1,3]);
                    bbox = get_boundingbox(pos, 0, cur_camera);

                    %% Estimate occlusion
                    pos = reshape(pos,[3,32]);
                    mono_pos = get_mono_coordinate(pos, cur_camera);
                    pos2d = reshape(cur_camera.project(pos'), [2, 32]);
                    mysk = Myskelmodel('full', ...
                                       reshape(mono_pos,[3,32]), ...
                                       cur_subject_model);
                    mono_ptc = mysk.generate_point_cloud();
                    n_pt = size(mono_ptc, 2);
                    mocap_ptc = get_mono_coordinate(mono_ptc, ...
                                                    cur_camera, 1);
                    mono2d_ptc = floor(reshape(cur_camera.project(mocap_ptc'),...
                                               [2,n_pt]));
                    %[occ, joint_takes, dummy] = ...
                    %    calculate_occlusion(double(mono_ptc), ...
                    %                        double(mono2d_ptc), ...
                    %                        double(mono_pos(:,mono_joint_list)),...
                    %                        double(pos2d(:,mono_joint_list)),...
                    %                        mysk.joint_maxdis, ...
                    %                        imeta.image_size(1), ...
                    %                        imeta.image_size(2),...
                    %                        mysk.occ_search_wsize(1), ...
                    %                        mysk.occ_search_wsize(2),...
                    %                        mysk.occ_ac_rate, ...
                    %                        masks);
                    %% Load Image
                    img = vobj.getFrame(fr);
                    if generate_type == 3
                        img(~fgmask_rgb) = 0;
                    end
                    % fprintf('%d\t', bbox);
                    % fprintf('\n');
                    bbox = max(1, bbox);
                    bbox(3) = min(bbox(3),size(img,2));
                    bbox(4) = min(bbox(4),size(img,1));
                    if bndadjustopt.ratio ~= 0
                        if bndadjustopt.ratio < 0
                            mask = logical(ones(size(img)));
                            mask(bbox(2):bbox(4),bbox(1):bbox(3),:) ...
                                = logical(0);
                            img(mask) = 0;
                        end
                        bbox = adjustbndbox(bbox, size(img), ...
                                            bndadjustopt);

                    end
                    %% Maybe redundant here, just for safety
                    %% consideration.  
                    bbox = max(1, bbox);
                    bbox(3) = min(bbox(3),size(img,2));
                    bbox(4) = min(bbox(4),size(img,1));
                    %% Be careful , bbox might be float number
                    bbox = round(bbox); %% added in Oct 4, 2014
                    
                    
                    if bbox(4) - bbox(2) > bbox(3) - bbox(1)
                        PAD = ((bbox(4) - bbox(2)) - (bbox(3) - bbox(1))) / 2;
                        bbox(3) = bbox(3) + PAD;
                        bbox(1) = bbox(1) - PAD;
                    else
                        PAD = ((bbox(3) - bbox(1)) - (bbox(4) - bbox(2))) / 2;
                        bbox(4) = bbox(4) + PAD;
                        bbox(2) = bbox(2) - PAD;
                    end
                    bbox = max(1, bbox);
                    bbox(3) = min(bbox(3),size(img,2));
                    bbox(4) = min(bbox(4),size(img,1));
                    %% Be careful , bbox might be float number
                    bbox = round(bbox); %% added in Oct 4, 2014
                    %%bbox = [1, 1, 1000, 1000];
                    
                    %% refine the occlusion indicator
                    %occ(imeta.mono_visible) = 0;
                    
                    pos2d_bnd = imcoor2bndcoor(pos2d(:,mono_joint_list), ...
                                               bbox, ...
                                               imeta ...
                                               .image_adjust_dim);
                    %% The whole structure 
                    % images_path, Y2d_bnd_body, oribbox,
                    cur_image_path = fullfile(subfolder_full, [subfolder,...
                                        sprintf('_%06d.jpg', fr)]);
                    %% Load image
                    
                    
                    cropped_img = imresize(img(bbox(2):bbox(4),bbox(1): ...
                                      bbox(3),:), [imeta.image_adjust_dim(1),...
                                           imeta.image_adjust_dim(2)]);
                    if generate_type == 3
                        cropped_mask = logical(imresize(fgmask(bbox(2):bbox(4),bbox(1): ...
                                      bbox(3)), [imeta.image_adjust_dim(1),...
                                           imeta.image_adjust_dim(2)]));
                    end
                    dimX = prod(imeta.image_adjust_dim([1,2]));
                    vimg = double(reshape(cropped_img, [dimX, 3]));
                    meancov = vimg' * vimg / dimX;
                    % Ensure all the image path have the same length
                    meta_data.bbox(:, fr) = bbox;
                    meta_data.images_path(1,fr) = {cur_image_path}; 
                    meta_data.Y3d_mono(:,fr) = ...
                        reshape(mono_pos(:,mono_joint_list), ...
                                [num_joint*3 , 1]);
                    meta_data.Y2d(:,fr) = pos2d_bnd(:);
                    %meta_data.occ_body(:,fr) = occ(:);
                    %meta_data.oribbox(:,fr) = bbox(:);
                    %meta_data.mean_image = meta_data.mean_image + ...
                    %    double(cropped_img);
                    %meta_data.rgb_meancov = meta_data.rgb_meancov + ...
                    %    meancov;
                    %% save cropped images

                    imwrite(cropped_img, cur_image_path);
                    
                    if generate_type == 3
                        cur_mask_path = fullfile(subfolder_mask_full, [subfolder, ...
                                            sprintf('_%06d.png', fr)]);
                        imwrite(cropped_mask, cur_mask_path);
                    end
                end
                %meta_data.mean_image = meta_data.mean_image / ...
                %    num_frames;
                %meta_data.rgb_meancov = meta_data.rgb_meancov/ num_frames;
                save(save_meta_name, '-struct', 'meta_data');
                fprintf('Finish %s in node %d under %s\n', vname, ...
                        node_id, subfolder);                    
            end
        end
    end
end

                                        
                    
                                        

