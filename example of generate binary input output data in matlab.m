% sensi_map: high quality sensi map

tmp_real = real(sensi_map);
tmp_img = imag(sensi_map);
tmp = cat(4, tmp_real, tmp_img);
fid = fopen(strcat('./path/to/training/label', num2str(subject_number), '.bin'), 'w');
fwrite(fid, tmp, 'float32');
fclose(fid);
