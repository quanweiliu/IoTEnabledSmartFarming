Channel_ID = 1754138;
ChannelReadKey = 'V5VKFGC13932RK0J' ;
ChannelWriteKey = 'FU593R8RJGXIQZD0' ;

% TO DO: Use the getWeatherUpdate function to get the temperature and
% humidity for the Townsville, Cairns, and Brisbane locations

[temperature1, humidity1] = getWeatherUpdate('townsville');

[temperature2, humidity2] = getWeatherUpdate('cairns');

[temperature3, humidity3] = getWeatherUpdate('brisbane');

thingSpeakWrite(Channel_ID, 'Fields', [1,2,3,4,5,6], 'Values' , [temperature1,humidity1,temperature2,humidity2,temperature3,humidity3], 'WriteKey', ChannelWriteKey);

[data, timestamps] = thingSpeakRead(Channel_ID, 'Fields', [1,2,3,4,5,6], 'NumPoints' , 10 , 'ReadKey', ChannelReadKey);

HUM_T = data(:,2);

HUM_C = data(:,4);

HUM_B = data(:,6);

% disp(actualTemperature)

plot(timestamps, HUM_T);
xlabel('timestamps')
ylabel('Humidity')
hold on;
plot(timestamps, HUM_C);
plot(timestamps, HUM_B);

legend('Humidity Townsville','Humidity Cairns','Humidity Brisbane')


% TO DO: Write the temperature and humidity values for each location to the
% six fields that you have created in your channel

% PROVIDED CODE STARTS HERE %
function [temperature, humidity] = getWeatherUpdate(stationName)
stationName = lower(stationName);

if strcmp(stationName, "townsville")
    url = 'http://www.bom.gov.au/fwo/IDQ60801/IDQ60801.94294.json';
elseif strcmp(stationName, "cairns")
    url = 'http://www.bom.gov.au/fwo/IDQ60801/IDQ60801.94287.json';
elseif strcmp(stationName, "brisbane")
    url = 'http://www.bom.gov.au/fwo/IDQ60901/IDQ60901.94576.json';
else
    error('Invalid station');
end

webdata = webread(url)

% Air temp
allTemperatures = vertcat(webdata.observations.data.air_temp);
temperature = allTemperatures(1);

% Relative humidity
allHumidities = vertcat(webdata.observations.data.rel_hum);
humidity = allHumidities(1);
end