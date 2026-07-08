require 'xcodeproj'

project_path = 'DarkAI.xcodeproj'
project = Xcodeproj::Project.open(project_path)
target = project.targets.find { |t| t.name == 'DarkAI' }

# 1. Add StableDiffusion.xcframework
framework_path = 'StableDiffusion.xcframework'
frameworks_group = project.groups.find { |g| g.display_name == 'Frameworks' } || project.main_group.new_group('Frameworks')
file_ref = frameworks_group.files.find { |f| f.path == framework_path }
unless file_ref
  file_ref = frameworks_group.new_file(framework_path)
end

frameworks_build_phase = target.frameworks_build_phase
unless frameworks_build_phase.files.any? { |f| f.file_ref && f.file_ref.path == framework_path }
  frameworks_build_phase.add_file_reference(file_ref)
end

# 3. Update Build Settings
target.build_configurations.each do |config|
  config.build_settings['SWIFT_OBJC_BRIDGING_HEADER'] = 'DarkAI/DarkAI-Bridging-Header.h'
  
  ldflags = config.build_settings['OTHER_LDFLAGS'] || ['$(inherited)']
  ldflags = [ldflags] if ldflags.is_a?(String)
  unless ldflags.include?('-lc++')
    ldflags << '-lc++'
  end
  config.build_settings['OTHER_LDFLAGS'] = ldflags
end

project.save
puts "Successfully updated Xcode project Build Settings and Frameworks!"
