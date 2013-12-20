function LOG(message, arg0, arg1)
  
  ts = datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');

  switch nargin
    case 1
      disp(sprintf('%s: %s', ts, message));
    case 2
      formatted = sprintf(message, arg0);
      disp(sprintf('%s: %s', ts, formatted));
    case 3
      formatted = sprintf(message, arg0, arg1);
      disp(sprintf('%s: %s', ts, formatted));
   otherwise
      error('LOGGING ERROR!')
  end

end
